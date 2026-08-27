from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings

from ...domain.document_identity import OWNER_SEPARATOR, parent_document_id
from ...domain.models import Document
from ...domain.repositories import VectorStoreRepo
from ..text.text_chunker import TextChunker


class FaissVectorStoreRepo(VectorStoreRepo):
    """In-memory vector store backed by FAISS (for local dev/tests without Postgres).

    Like the Postgres variant: it chunks documents before embedding and returns the real
    parent document id from the chunk metadata. State is held in process memory — it is
    gone after a restart.
    """

    def __init__(self, embeddings: Embeddings, chunker: TextChunker | None = None) -> None:
        self._embeddings = embeddings
        self._chunker = chunker or TextChunker()
        self._vector_store: FAISS | None = None
        # Maps (owner_id, document id) -> list of its chunk ids, used for deletion. The
        # owner is part of the key so a delete cannot reach another user's chunks, matching
        # what the Postgres and Neo4j adapters enforce in their queries.
        self._chunk_ids_by_owned_document: dict[tuple[str, str], list[str]] = {}

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        # Stamp the owner onto every document before chunking (chunks inherit the metadata
        # → owner_id lands on every vector, for the retrieval filter).
        owned_documents = [
            Document(
                id=document.id,
                content=document.content,
                metadata={**document.metadata, "owner_id": owner_id},
            )
            for document in documents
        ]

        # Re-upload: drop the existing chunks of these documents before adding new ones, so
        # none are orphaned (when the new version has fewer chunks).
        for existing_document_id in {document.id for document in owned_documents}:
            await self.delete_by_document_id(existing_document_id, owner_id)

        chunks = self._chunker.chunk_many(owned_documents)
        if not chunks:
            return

        langchain_documents = [
            LangChainDocument(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]
        # The chunker numbers chunks per document ("{doc_id}#{n}"), which is unique only as
        # long as document ids are. FAISS refuses duplicate ids outright, so the owner goes
        # into the key here — two users uploading the same document id stay separate.
        chunk_ids = [f"{owner_id}{OWNER_SEPARATOR}{chunk.id}" for chunk in chunks]

        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(
                langchain_documents, self._embeddings, ids=chunk_ids
            )
        else:
            self._vector_store.add_documents(langchain_documents, ids=chunk_ids)

        for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
            key = (owner_id, chunk.metadata["doc_id"])
            self._chunk_ids_by_owned_document.setdefault(key, []).append(chunk_id)

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        if self._vector_store is None:
            return []

        # The metadata filter limits results to the asker's own chunks. fetch_k > k because
        # the filter is applied after fetching — otherwise we would return too few results.
        results = self._vector_store.similarity_search(
            query,
            k=top_k,
            filter={"owner_id": owner_id},
            fetch_k=max(top_k * 5, 50),
        )
        return [
            Document(
                id=parent_document_id(result.metadata),
                content=result.page_content,
                metadata=result.metadata,
            )
            for result in results
        ]

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        if self._vector_store is None:
            return

        chunk_ids = self._chunk_ids_by_owned_document.pop((owner_id, doc_id), [])
        if chunk_ids:
            self._vector_store.delete(chunk_ids)
