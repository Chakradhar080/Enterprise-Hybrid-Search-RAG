import faiss
from langchain_core import documents
import numpy as np
import pickle
import faiss
import numpy as np
from pymupdf.mupdf import vectors

from app.models.document import Document


class FAISSVectorStore:

    def __init__(self):
        self.index = None
        self.documents = []

    def build_index(self, documents: list[Document]):
        embeddings = np.array(
            [doc.embedding for doc in documents],
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        self.documents = documents

    def search(self, query_embedding, k=3):
        query = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(query, k)

        return [self.documents[i] for i in indices[0]]
    
    def save(self, index_path, metadata_path):

        faiss.write_index(
            self.index,
            str(index_path)
            )

        with open(metadata_path, "wb") as f:
            pickle.dump(
                self.documents,
                f
            )

    def load(self, index_path, metadata_path):

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(metadata_path, "rb") as f:
            self.documents = pickle.load(f)
            
    def add_documents(self, documents):
        vectors = np.array(
            [doc.embedding for doc in documents],
            dtype="float32"
            )

        self.index.add(vectors)

        self.documents.extend(documents)