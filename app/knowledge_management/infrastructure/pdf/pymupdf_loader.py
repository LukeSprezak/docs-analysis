import pymupdf4llm
from typing import List
from ...domain.models import Document

class PyMuPDFLoader:
    def load_pdf(self, file_path: str) -> List[Document]:
        md_text = pymupdf4llm.to_markdown(file_path)
        return [Document(id=file_path, content=md_text, metadata={"source": file_path})]
