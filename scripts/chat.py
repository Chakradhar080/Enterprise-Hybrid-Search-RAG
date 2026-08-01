from pathlib import Path

from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.prompts.prompt_builder import PromptBuilder
from app.llm.ollama_client import OllamaClient

VECTOR_DIR = Path("vector_store")

# Load FAISS
store = FAISSVectorStore()
store.load(
    VECTOR_DIR / "faiss.index",
    VECTOR_DIR / "metadata.pkl"
)

# Rebuild BM25 from stored documents
bm25 = BM25Retriever()
bm25.build_index(store.documents)

hybrid = HybridRetriever(store, bm25)

embedding_model = EmbeddingModel()

prompt_builder = PromptBuilder()

llm = OllamaClient()

print("=" * 60)
print(" Hybrid RAG Chat ")
print("=" * 60)

while True:

    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    query_embedding = embedding_model.embed_query(query)

    documents = hybrid.search(
        query,
        query_embedding,
        k=5
    )

    prompt = prompt_builder.build(
        query,
        documents
    )

    answer = llm.generate(prompt)

    print("\nAI:\n")
    print(answer)