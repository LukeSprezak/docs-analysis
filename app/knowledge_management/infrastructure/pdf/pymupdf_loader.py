import pymupdf4llm

from ...domain.models import Document


class PyMuPDFLoader:
    def load_pdf(self, file_path: str) -> list[Document]:
        # page_chunks=True zwraca jeden wpis na stronę, dzięki czemu możemy
        # zachować numer strony w metadanych (do cytowań w odpowiedziach).
        pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)

        documents: list[Document] = []
        for page_number, page in enumerate(pages, start=1):
            text = page.get("text", "") if isinstance(page, dict) else str(page)
            if not text.strip():
                continue
            documents.append(
                Document(
                    id=file_path,
                    content=text,
                    metadata={"source": file_path, "page": page_number},
                )
            )
        return documents