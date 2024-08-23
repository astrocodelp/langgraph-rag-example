import fs from 'fs';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { type DocumentInterface } from '@langchain/core/documents';
import { TavilySearchAPIRetriever } from '@langchain/community/retrievers/tavily_search_api';
import { END, START, StateGraph } from '@langchain/langgraph';
import { Annotation } from '@langchain/langgraph';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatOpenAI } from '@langchain/openai';
import { formatDocumentsAsString } from 'langchain/util/document';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { pull } from 'langchain/hub';
import type { RunnableConfig } from '@langchain/core/runnables';

// Sample documents for the vector store
const documents: DocumentInterface[] = [
  {
    pageContent:
      'JavaScript is a versatile programming language primarily used for web development.',
    metadata: {
      id: '1',
    },
  },
  {
    pageContent:
      'Langchain is a powerful library for building language model applications.',
    metadata: {
      id: '2',
    },
  },
  {
    pageContent:
      'Retrieval-Augmented Generation combines retrieval-based and generative models.',
    metadata: {
      id: '3',
    },
  },
  {
    pageContent:
      'Langsmith is a tool that aids in the development and debugging of language model applications.',
    metadata: {
      id: '4',
    },
  },
];

// Create embeddings from the documents
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'text-embedding-3-small',
});

// Create a new vector store
const vectorStore = new MemoryVectorStore(embeddings);

// Add the documents in the vector store
await vectorStore.addDocuments(documents);

// Represents the state of our graph.
const GraphState = Annotation.Root({
  documents: Annotation<DocumentInterface[]>({
    reducer: (x, y) => (y ? y.concat(x ?? []) : []),
  }),
  question: Annotation<string>({
    reducer: (x, y) => y ?? x ?? '',
  }),
  generation: Annotation<string>({
    reducer: (x, y) => y ?? x,
  }),
});

import { ScoreThresholdRetriever } from 'langchain/retrievers/score_threshold';
/**
 * Retrieve documents
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function retrieve(
  state: typeof GraphState.State,
  config?: RunnableConfig
): Promise<Partial<typeof GraphState.State>> {
  console.log('---RETRIEVE---');

  const retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore, {
    minSimilarityScore: 0.3, // Finds results with at least this similarity score
    maxK: 1, // Maximum number of results to return
    kIncrement: 1, // Increment the number of results by this amount
  });

  const relatedDocuments = await retriever
    // Optional: Set the run name for tracing - useful for debugging
    .withConfig({ runName: 'FetchRelevantDocuments' })
    .invoke(state.question, config);

  return {
    documents: relatedDocuments,
  };
}

/**
 * Web search based on the question using Tavily API.
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function webSearch(
  state: typeof GraphState.State,
  config?: RunnableConfig
): Promise<Partial<typeof GraphState.State>> {
  console.log('---WEB SEARCH---');

  const retriever = new TavilySearchAPIRetriever({
    apiKey: process.env.TAVILY_API_KEY,
    k: 1,
  });

  const webDocuments = await retriever
    // Optional: Set the run name for tracing - useful for debugging
    .withConfig({ runName: 'FetchRelevantDocuments' })
    .invoke(state.question, config);

  return {
    documents: webDocuments,
  };
}

/**
 * Generate answer
 *
 * @param {typeof GraphState.State} state The current state of the graph.
 * @param {RunnableConfig | undefined} config The configuration object for tracing.
 * @returns {Promise<Partial<typeof GraphState.State>>} The new state object.
 */
async function generate(
  state: typeof GraphState.State,
  config?: RunnableConfig
): Promise<Partial<typeof GraphState.State>> {
  console.log('---GENERATE---');

  // Define the LLM
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o-mini',
    temperature: 0,
  });

  // Pull the RAG prompt from the hub - https://smith.langchain.com/hub/rlm/rag-prompt
  const prompt = await pull<ChatPromptTemplate>('rlm/rag-prompt');
  // Construct the RAG chain by piping the prompt, model, and output parser
  const ragChain = prompt.pipe(model).pipe(new StringOutputParser());

  const generation = await ragChain
    // Optional: Set the run name for tracing - useful for debugging
    .withConfig({ runName: 'GenerateAnswer' })
    .invoke(
      {
        context: formatDocumentsAsString(state.documents),
        question: state.question,
      },
      config
    );

  return {
    generation,
  };
}

// Define the workflow and add the nodes
const workflow = new StateGraph(GraphState)
  .addNode('retrieve', retrieve)
  .addNode('webSearch', webSearch)
  .addNode('generate', generate);

// Define the edges
workflow.addEdge(START, 'retrieve');
// - If no documents are retrieved, go to web search
// - If documents are retrieved, go to generate
workflow.addConditionalEdges(
  'retrieve',
  (state: typeof GraphState.State) =>
    state.documents.length === 0 ? 'webSearch' : 'generate',
  {
    webSearch: 'webSearch',
    generate: 'generate',
  }
);
workflow.addEdge('webSearch', 'generate');
workflow.addEdge('generate', END);

const app = workflow.compile();

// Visualize the graph
const graphPng = await app.getGraph().drawMermaidPng();
const buffer = Buffer.from(await graphPng.arrayBuffer());
// Save the graph to a file
fs.writeFileSync('graph.png', buffer);

// Invoke the graph
const question = process.argv[2] ?? 'What is Langchain?'; // Get the question from the command line
const output = await app.invoke({ question }); // Invoke the graph with the question

console.log(output);
