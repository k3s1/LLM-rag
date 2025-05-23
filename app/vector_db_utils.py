# app/vector_db_utils.py
import os
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings # Using BGE for local embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings

# Initialize Embedding Model
# For local usage, 'BAAI/bge-small-en-v1.5' is a good balance of performance and size.
# You might need to download it the first time, which can take a moment.
try:
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    print(f"Error initializing HuggingFaceBgeEmbeddings: {e}")
    print("Falling back to a different embedding model or check your internet connection for model download.")
    # Fallback or alternative if BGE fails (e.g., if you plan to use OpenAI/Gemini embeddings)
    # For this guide, we'll assume BGE works.

def load_document(file_path: str) -> List[Document]:
    """Loads a document using Langchain's PyPDFLoader."""
    # [cite: 4]
    try:
        loader = PyPDFLoader(file_path)
        # [cite: 4]
        documents = loader.load()
        # [cite: 4]
        # Limit pages for now if necessary, though PyPDFLoader handles it.
        # If you want to strictly enforce MAX_PAGES_PER_DOCUMENT here, you'd iterate and count.
        if len(documents) > settings.MAX_PAGES_PER_DOCUMENT:
            print(f"Warning: Document '{file_path}' exceeds {settings.MAX_PAGES_PER_DOCUMENT} pages. Truncating.")
            documents = documents[:settings.MAX_PAGES_PER_DOCUMENT]
        return documents
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return []

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunks documents into manageable sizes."""
    # [cite: 5]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, # [cite: 5]
        chunk_overlap=settings.CHUNK_OVERLAP # [cite: 5]
    )
    chunks = text_splitter.split_documents(documents) # [cite: 5]
    return chunks

def get_vector_store(chunks: List[Document], faiss_index: FAISS = None) -> FAISS:
    """
    Generates embeddings for chunks and stores them in a FAISS vector database.
    If an existing FAISS index is provided, it's updated.
    """
    # [cite: 6]
    if not embeddings:
        raise RuntimeError("Embedding model not initialized.")

    if faiss_index:
        # [cite: 6]
        print("Adding new chunks to existing FAISS index.")
        faiss_index.add_documents(chunks)
    else:
        # [cite: 6]
        print("Creating new FAISS index from chunks.")
        faiss_index = FAISS.from_documents(chunks, embeddings)
    return faiss_index

def save_faiss_index(faiss_index: FAISS, path: str):
    """Saves the FAISS index to disk."""
    faiss_index.save_local(path)
    print(f"FAISS index saved to {path}")

def load_faiss_index(path: str) -> FAISS | None:
    """Loads the FAISS index from disk."""
    if os.path.exists(path + "/index.faiss") and os.path.exists(path + "/index.pkl"):
        try:
            # [cite: 6]
            faiss_index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            print(f"FAISS index loaded from {path}")
            return faiss_index
        except Exception as e:
            print(f"Error loading FAISS index from {path}: {e}")
            return None
    print(f"No FAISS index found at {path}")
    return None

def retrieve_documents(query: str, faiss_index: FAISS, k: int = 4) -> List[Document]:
    """Retrieves relevant document chunks from the FAISS index."""
    # [cite: 7]
    if not embeddings:
        raise RuntimeError("Embedding model not initialized.")
    if not faiss_index:
        raise ValueError("FAISS index not loaded. Please upload documents first.")

    # [cite: 7]
    docs = faiss_index.similarity_search(query, k=k)
    return docs