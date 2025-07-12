# LangChain Bot

A simple chatbot application.

## Getting Started

1. **Start the Application**

    ```bash
    docker-compose up
    ```

2. **Access the UI**

    Open your browser and go to: [http://localhost:8501](http://localhost:8501)

3. **Upload a Document**

    Use the Streamlit interface to upload your document.

4. **Process the Document**

    - The document will be chunked into smaller parts.
    - Each chunk will be embedded for semantic understanding.
    - The embeddings are stored in the database.

5. **Chat with the Bot**

    Interact with the chatbot to ask questions about the document you just uploaded.
