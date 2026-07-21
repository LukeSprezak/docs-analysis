from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings

from ...domain.models import Document
from ...domain.repositories import VectorStoreRepo
from ..text.text_chunker import TextChunker


class FaissVectorStoreRepo(VectorStoreRepo):
    """Magazyn wektorowy w pamięci oparty o FAISS (do lokalnego dev/testów bez Postgresa).

    Tak jak wariant Postgresowy: fragmentuje dokumenty przed embedowaniem i zwraca
    prawdziwe id dokumentu nadrzędnego z metadanych fragmentu. Stan trzymany jest w
    pamięci procesu — znika po restarcie.
    """

    def __init__(self, embeddings: Embeddings, chunker: TextChunker | None = None) -> None:
        self._embeddings = embeddings
        self._chunker = chunker or TextChunker()
        self._vector_store: FAISS | None = None
        # Mapowanie id dokumentu (filename) -> lista id jego fragmentów, do usuwania.
        self._chunk_ids_by_document_id: dict[str, list[str]] = {}

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        # Stempel właściciela na każdym dokumencie przed chunkingiem (fragmenty
        # dziedziczą metadane → owner_id w każdym wektorze, do filtra retrievalu).
        owned_documents = [
            Document(
                id=document.id,
                content=document.content,
                metadata={**document.metadata, "owner_id": owner_id},
            )
            for document in documents
        ]

        # Re-upload: usuń istniejące fragmenty tych dokumentów przed dodaniem nowych,
        # żeby nie zostawały osierocone (gdy nowa wersja ma mniej fragmentów).
        for existing_document_id in {document.id for document in owned_documents}:
            await self.delete_by_document_id(existing_document_id, owner_id)

        chunks = self._chunker.chunk_many(owned_documents)
        if not chunks:
            return

        langchain_documents = [
            LangChainDocument(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]
        chunk_ids = [chunk.id for chunk in chunks]

        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(
                langchain_documents, self._embeddings, ids=chunk_ids
            )
        else:
            self._vector_store.add_documents(langchain_documents, ids=chunk_ids)

        for chunk in chunks:
            # doc_id jest namespace'owany per user (id dokumentu nadrzędnego) → unikalny.
            document_id = chunk.metadata.get("doc_id") or chunk.metadata.get("filename")
            if document_id is not None:
                self._chunk_ids_by_document_id.setdefault(document_id, []).append(chunk.id)

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        if self._vector_store is None:
            return []

        # Filtr po metadanych ogranicza wyniki do fragmentów pytającego. fetch_k > k,
        # bo filtr odsiewa po pobraniu — inaczej zwrócilibyśmy za mało wyników.
        results = self._vector_store.similarity_search(
            query,
            k=top_k,
            filter={"owner_id": owner_id},
            fetch_k=max(top_k * 5, 50),
        )
        return [
            Document(
                id=str(
                    result.metadata.get("doc_id")
                    or result.metadata.get("filename")
                    or "unknown"
                ),
                content=result.page_content,
                metadata=result.metadata,
            )
            for result in results
        ]

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        if self._vector_store is None:
            return

        # doc_id jest namespace'owany per user, więc sam w sobie wystarcza do izolacji.
        chunk_ids = self._chunk_ids_by_document_id.pop(doc_id, [])
        if chunk_ids:
            self._vector_store.delete(chunk_ids)