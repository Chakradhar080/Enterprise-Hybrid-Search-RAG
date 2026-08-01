from pathlib import Path

from fastapi import FastAPI

from app.api import routes
from app.api.routes import router

from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from app.services.rag_service import RAGService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.memory_service import MemoryService

# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

VECTOR_DIR = Path("vector_store")

store = FAISSVectorStore()

store.load(
    VECTOR_DIR / "faiss.index",
    VECTOR_DIR / "metadata.pkl"
)

# --------------------------------------------------
# Build BM25
# --------------------------------------------------

bm25 = BM25Retriever()

bm25.build_index(store.documents)

# --------------------------------------------------
# Hybrid Retriever
# --------------------------------------------------

hybrid = HybridRetriever(
    store,
    bm25
)

# --------------------------------------------------
# Initialize Services
# --------------------------------------------------

routes.rag_service = RAGService(hybrid)

routes.document_service = DocumentService(store)

routes.ingestion_service = IngestionService(
    store,
    bm25
)

routes.memory_service = MemoryService()

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Hybrid Search RAG API",
    version="1.0.0",
    description="Enterprise Hybrid Search RAG over Internal Documents"
)

app.include_router(router)