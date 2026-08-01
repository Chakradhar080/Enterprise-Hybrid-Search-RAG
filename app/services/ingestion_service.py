from pathlib import Path

from app.loaders.pdf_loader import PDFLoader
from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_model import EmbeddingModel


class IngestionService:

    def __init__(self, store, bm25):

        self.store = store
        self.bm25 = bm25

        self.chunker = TextChunker()

        self.embedding = EmbeddingModel()

    def ingest(self, pdf_folder):

        loader = PDFLoader(pdf_folder)

        documents = loader.load_documents()

        chunks = self.chunker.chunk_documents(documents)

        for chunk in chunks:
            chunk.embedding = self.embedding.embed_query(chunk.text)

        self.store.documents.extend(chunks)

        self.store.add_documents(chunks)

        self.bm25.build_index(self.store.documents)

        return len(chunks)