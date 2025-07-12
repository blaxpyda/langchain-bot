from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import os

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def embed_text(text):
    """Embeds text using Cohere embedding function."""
    # Create Cohere embedding function
    cohere_ef  = embedding_functions.CohereEmbeddingFunction(api_key=os.getenv("COHERE_API_KEY"),  model_name="large")
  # If text is a string, convert to list for the embedding function
    if isinstance(text, str):
        return cohere_ef([text])[0]
    else:
        return cohere_ef(text)
