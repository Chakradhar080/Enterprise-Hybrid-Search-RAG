from pathlib import Path
import fitz

from app.models.document import Document
from app.preprocess.cleaner import TextCleaner

class PDFLoader:

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_documents(self):

        documents = []

        pdf_files = self.data_path.rglob("*.pdf")

        for pdf in pdf_files:

            folder_name = pdf.parent.name
            file_name = pdf.name

            document = fitz.open(pdf)

            for page_number, page in enumerate(document):

                text = TextCleaner.clean(page.get_text())

                if text.strip():

                    documents.append(
                        Document(
                            text=text,
                            file_name=file_name,
                            folder=folder_name,
                            page_number=page_number + 1,
                        )
                    )

        return documents