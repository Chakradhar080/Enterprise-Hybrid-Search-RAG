from collections import defaultdict

from app.models.document import Document


class HybridRetriever:

    def __init__(self, faiss_store, bm25):

        self.faiss_store = faiss_store
        self.bm25 = bm25

    def search(
        self,
        query,
        query_embedding,
        k=5,
        rrf_k=60
    ):

        faiss_results = self.faiss_store.search(
            query_embedding,
            k=k
        )

        bm25_results = self.bm25.search(
            query,
            k=k
        )

        scores = defaultdict(float)
        documents = {}

        for rank, doc in enumerate(faiss_results):

            key = (
                doc.file_name,
                doc.page_number,
                doc.text
            )

            scores[key] += 1 / (rrf_k + rank + 1)

            documents[key] = doc

        for rank, doc in enumerate(bm25_results):

            key = (
                doc.file_name,
                doc.page_number,
                doc.text
            )

            scores[key] += 1 / (rrf_k + rank + 1)

            documents[key] = doc

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            documents[key]
            for key, _ in ranked[:k]
        ]