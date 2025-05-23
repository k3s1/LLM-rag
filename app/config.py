# app/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Settings:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini") # Default to Gemini
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # Database settings
    DATABASE_URL: str = "sqlite:///./data/documents.db" # SQLite for simplicity

    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENTS: int = 20
    MAX_PAGES_PER_DOCUMENT: int = 1000

    # Paths
    DOCUMENTS_DIR: str = "documents"
    FAISS_INDEX_PATH: str = "data/faiss_index" # Path to save/load FAISS index

settings = Settings()

# Create necessary directories if they don't exist
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)