# RAG System with Langchain, LangGraph, and Tavily in TypeScript

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using TypeScript, leveraging Langchain, LangGraph, LangSmith, and Tavily. The RAG system retrieves relevant documents and generates contextually appropriate responses.

## Prerequisites

- **Node.js**: Make sure you have Node.js installed on your machine.
- **pnpm**: This project uses `pnpm` as the package manager. You can install it by running:

  ```bash
  npm install -g pnpm
  ```

## Installation

1. **Clone the Repository**:
   
   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/astrocodelp/langgraph-rag-example.git
   ```

   Navigate into the project directory:

   ```bash
   cd langgraph-rag-example
   ```

2. **Install Dependencies**:

   Install the necessary dependencies using `pnpm`:

   ```bash
   pnpm install
   ```

3. **Set Up Environment Variables**:

   Create a `.env` file in the root of your project and add your OpenAI and Tavily API keys:

   ```plaintext
   OPENAI_API_KEY=your-openai-api-key
   TAVILY_API_KEY=your-tavily-api-key
   ```

   Make sure you replace `your-openai-api-key` and `your-tavily-api-key` with your actual API keys.

   If you want to trace the RAG using LangSmith and LangGraph, you can also add the following environment variables:

   ```plaintext
    LANGCHAIN_TRACING_V2=true
    LANGSMITH_API_KEY="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"
    LANGCHAIN_PROJECT="langgraph-rag-demo"

    # Optional: Enable background callbacks for tracing - Use it when you are not using a serverless environment
    LANGCHAIN_CALLBACKS_BACKGROUND=true
   ```

## Running the Project

To run the project, execute the following command:

```bash
pnpm start "What is Langchain?"
```

This command will run the `graph.ts` file, which sets up and executes the RAG system.

### Expected Output

If everything is set up correctly, you should see output similar to the following:

```bash
> tsx -r dotenv/config graph.ts

---RETRIEVE---
---GENERATE---
{
  documents: [
    Document {
      pageContent: 'Langchain is a powerful library for building language model applications.',
      metadata: [Object],
      id: undefined
    }
  ],
  question: 'What is Langchain?',
  generation: 'Langchain is a powerful library designed for creating applications that utilize language models. It provides tools and frameworks to facilitate the development of these applications.'
}
```

## Project Structure

- **`graph.ts`**: The main file where the RAG system is defined and executed.
- **`graph.png`**: A diagram illustrating the RAG system.
- **`.env`**: Environment variables file for storing API keys.
- **`package.json`**: Project configuration, including dependencies and scripts.

## Customization

You can extend this project by:

- Adding more documents to the vector store.
- Changing the embeddings model.
- Integrating additional retrievers or modifying the existing ones.
- Deploying the system as an API or a service.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
