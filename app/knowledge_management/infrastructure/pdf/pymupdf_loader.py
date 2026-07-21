import pymupdf4llm

from ...domain.models import Document


class PyMuPDFLoader:
    def load_pdf(self, file_path: str) -> list[Document]:
        # page_chunks=True returns one entry per page, which lets us keep the page number
        # in the metadata (for citations in answers).
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
