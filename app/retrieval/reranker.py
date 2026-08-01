from sentence_transformers import CrossEncoder
from app.models.document import Document


class Reranker:

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5):

        pairs = [(query, doc.text) for doc in documents]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        print("\n========== RERANKER SCORES ==========\n")

        for score, doc in ranked:
            print(f"{score:.4f}  -->  {doc.file_name}")

        print("\n=====================================\n")

        return [doc for score, doc in ranked[:top_k]]