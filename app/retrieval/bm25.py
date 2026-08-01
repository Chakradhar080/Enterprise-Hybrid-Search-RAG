from rank_bm25 import BM25Okapi

from app.models.document import Document


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.bm25 = None

    def build_index(self, documents: list[Document]):

        self.documents = documents

        tokenized_documents = [
            document.text.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query: str, k: int = 3):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self.documents),
            reverse=True,
            key=lambda x: x[0]
        )

        return [
            document
            for _, document in ranked[:k]
        ]