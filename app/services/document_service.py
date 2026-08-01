from app.vectorstore.faiss_store import FAISSVectorStore


class DocumentService:

    def __init__(self, store):
        self.store = store

    def get_documents(self):

        documents = []
        seen = set()

        for doc in self.store.documents:

            key = (doc.folder, doc.file_name)

            if key not in seen:

                seen.add(key)

                documents.append({
                    "folder": doc.folder,
                    "file": doc.file_name,
                    "page": doc.page_number
                })

        documents.sort(
            key=lambda x: (x["folder"], x["file"])
        )

        return documents