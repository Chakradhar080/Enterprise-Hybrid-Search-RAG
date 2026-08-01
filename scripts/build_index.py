from pathlib import Path

from app.config import RAW_DATA_DIR
from app.loaders.pdf_loader import PDFLoader
from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.faiss_store import FAISSVectorStore

VECTOR_DIR = Path("vector_store")
VECTOR_DIR.mkdir(exist_ok=True)

loader = PDFLoader(RAW_DATA_DIR)
documents = loader.load_documents()

chunker = TextChunker()
chunks = chunker.chunk_documents(documents)

embedding_model = EmbeddingModel()

for chunk in chunks:
    chunk.embedding = embedding_model.embed_query(chunk.text)

store = FAISSVectorStore()
store.build_index(chunks)

store.save(
    VECTOR_DIR / "faiss.index",
    VECTOR_DIR / "metadata.pkl"
)

print("Index saved successfully!")