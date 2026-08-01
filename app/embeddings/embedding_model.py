from langchain_ollama import OllamaEmbeddings

from app.config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

    def embed_query(self, text: str):
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: list[str]):
        return self.embedding_model.embed_documents(texts)