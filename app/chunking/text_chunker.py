from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import Document


class TextChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

    def chunk_documents(self, documents: list[Document]):

        chunked_documents = []

        chunk_id = 1

        for document in documents:

            chunks = self.splitter.split_text(document.text)

            for chunk in chunks:

                metadata = {
                    "source": document.file_name,
                    "folder": document.folder,
                    "page": document.page_number,
                    "chunk_id": chunk_id,
                }

                chunked_documents.append(
                    Document(
                        text=chunk,
                        file_name=document.file_name,
                        folder=document.folder,
                        page_number=document.page_number,
                        metadata=metadata,
                    )
                )

                chunk_id += 1

        return chunked_documents