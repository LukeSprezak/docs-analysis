from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...domain.models import Document


class TextChunker:
    """Dzieli dokumenty na mniejsze fragmenty pod retrieval.

    Każdy fragment dziedziczy metadane dokumentu źródłowego i dostaje
    dodatkowo ``doc_id`` (id dokumentu nadrzędnego) oraz ``chunk_index``,
    dzięki czemu można je później zidentyfikować i zacytować.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Dzieli listę dokumentów na fragmenty z globalnie unikalnymi id.

        ``documents`` to zwykle strony jednego pliku (albo pojedynczy
        dokument dla plików tekstowych). Indeks fragmentu jest wspólny dla
        całego pliku, więc id ``"{doc_id}::{n}"`` nie kolidują między stronami.
        """
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
