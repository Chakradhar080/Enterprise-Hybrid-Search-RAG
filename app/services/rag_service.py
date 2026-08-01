from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.reranker import Reranker
from app.prompts.prompt_builder import PromptBuilder
from app.llm.ollama_client import OllamaClient
from app.models.chat_models import ChatResponse


class RAGService:

    def __init__(self, hybrid):

        self.hybrid = hybrid
        self.embedding = EmbeddingModel()
        self.reranker = Reranker()
        self.prompt_builder = PromptBuilder()
        self.llm = OllamaClient()

    # --------------------------------------------------
    # Standard Chat
    # --------------------------------------------------

    def ask(
        self,
        query: str,
        history=None
    ) -> ChatResponse:

        query_embedding = self.embedding.embed_query(query)

        documents = self.hybrid.search(
            query=query,
            query_embedding=query_embedding,
            k=10
        )

        documents = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=5
        )

        prompt = self.prompt_builder.build(
            query=query,
            documents=documents,
            history=history
        )

        answer = self.llm.generate(prompt)

        sources = [
            {
                "file": doc.file_name,
                "page": doc.page_number
            }
            for doc in documents
        ]

        return ChatResponse(
            answer=answer,
            sources=sources
        )

    # --------------------------------------------------
    # Streaming Chat
    # --------------------------------------------------

    def stream(
        self,
        query: str,
        history=None
    ):

        # Retrieve Documents

        query_embedding = self.embedding.embed_query(query)

        documents = self.hybrid.search(
            query=query,
            query_embedding=query_embedding,
            k=10
        )

        documents = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=5
        )

        # Build Prompt

        prompt = self.prompt_builder.build(
            query=query,
            documents=documents,
            history=history
        )

        # Build Sources

        sources = [
            {
                "file": doc.file_name,
                "page": doc.page_number
            }
            for doc in documents
        ]

        # Stream Tokens

        for token in self.llm.stream(prompt):

            yield {
                "type": "token",
                "content": token
            }

        # Send Sources

        yield {
            "type": "sources",
            "content": sources
        }

        # Notify Completion

        yield {
            "type": "done"
        }