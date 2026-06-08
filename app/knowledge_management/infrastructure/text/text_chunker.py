from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...domain.models import Document


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        counter = 0
        for document in documents:
            for text in self._splitter.split_text(document.content):
                if not text.strip():
                    continue
                metadata = {
                    **document.metadata,
                    "doc_id": document.id,
                    "chunk_index": counter,
                }
                chunks.append(
                    Document(id=f"{document.id}::{counter}", content=text, metadata=metadata)
                )
                counter += 1
        return chunks
