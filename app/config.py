from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

# Vector Store
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Ollama Models
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100