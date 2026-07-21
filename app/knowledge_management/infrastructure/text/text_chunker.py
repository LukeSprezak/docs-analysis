from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...domain.models import Document


class TextChunker:
    """Splits documents into smaller chunks for retrieval.

    Every chunk inherits the metadata of its source document and additionally receives
    ``doc_id`` (the parent document id) and ``chunk_index``, so it can later be identified
    and cited.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Splits a list of documents into chunks with globally unique ids.

        ``documents`` is usually the pages of a single file (or one document for text
        files). The chunk index is shared across the whole file, so ``"{doc_id}::{n}"`` ids
        never collide between pages.
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
