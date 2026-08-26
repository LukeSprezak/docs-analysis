import logging
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.shared.database import db_connection, get_engine

from ...application.retrieval.rank_fusion import fuse_documents, retrieval_key
from ...domain.models import Document
from ...domain.repositories import VectorStoreRepo
from ..text.text_chunker import TextChunker

logger = logging.getLogger(__name__)


class PostgresVectorStoreRepo(VectorStoreRepo):
    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "documents",
        chunker: TextChunker | None = None,
        enable_hybrid_search: bool = False,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.chunker = chunker or TextChunker()
        self.enable_hybrid_search = enable_hybrid_search

        self.vector_store = PGVector(
            connection=get_engine(),
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            use_jsonb=True,
            async_mode=True,
            create_extension=False,
        )

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        owned_documents = [
            Document(
                id=document.id,
                content=document.content,
                metadata={**document.metadata, "owner_id": owner_id},
            )
            for document in documents
        ]

        # Re-upload: delete the existing chunks of these documents BEFORE inserting new ones.
        # Without this, when the new version has fewer chunks, the old ones with higher
        # indices are orphaned in the database and pollute retrieval.
        for document_id in {document.id for document in owned_documents}:
            await self.delete_by_document_id(document_id, owner_id)

        # Documents are split into chunks BEFORE embedding. Without this the whole file would
        # land in the database as a single vector and retrieval would be useless.
        chunks = self.chunker.chunk_many(owned_documents)
        if not chunks:
            return

        lc_docs = [
            LCDocument(page_content=chunk.content, metadata=chunk.metadata or {})
            for chunk in chunks
        ]
        ids = [chunk.id for chunk in chunks]
        await self.vector_store.aadd_documents(lc_docs, ids=ids)

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        if self.enable_hybrid_search:
            return await self._hybrid_search(query, owner_id, top_k)
        return await self._vector_search(query, owner_id, top_k)

    async def _vector_search(self, query: str, owner_id: str, top_k: int) -> list[Document]:
        # Metadata filter: return only the chunks belonging to the asker.
        results = await self.vector_store.asimilarity_search(
            query, k=top_k, filter={"owner_id": {"$eq": owner_id}}
        )
        return [
            Document(
                # The real parent document id from the chunk metadata (PGVector does not
                # expose .id on the result → this used to come out as "unknown").
                id=str(res.metadata.get("doc_id") or res.metadata.get("filename") or "unknown"),
                content=res.page_content,
                metadata=res.metadata,
            )
            for res in results
        ]

    async def _hybrid_search(self, query: str, owner_id: str, top_k: int) -> list[Document]:
        """Combines vector and full-text search (Postgres FTS) via RRF.

        Each method fetches ``top_k`` candidates and the fusion returns the best ``top_k``
        after merging — a chunk relevant in both rankings is promoted. Full-text search adds
        what vectors miss: matching on keywords, proper nouns and specific terms.
        """
        vector_documents = await self._vector_search(query, owner_id, top_k)
        keyword_documents = await self._keyword_search(query, owner_id, top_k)
        return fuse_documents([vector_documents, keyword_documents], top_k=top_k, key_of=retrieval_key)

    async def _keyword_search(self, query: str, owner_id: str, top_k: int) -> list[Document]:
        """Full-text search over chunk content (Postgres FTS, ranked with ts_rank).

        The 'simple' configuration (no language-dependent stemmer) is portable and works
        sensibly for Polish documents. Production note: on a large database add a GIN index on
        ``to_tsvector('simple', document)`` (an Alembic task) — here it is computed on the fly.
        """
        async with db_connection() as connection:
            collection_uuid = await self._collection_uuid(connection)
            if collection_uuid is None:
                return []
            result = await connection.execute(
                text(
                    "SELECT document, cmetadata "
                    "FROM langchain_pg_embedding "
                    "WHERE collection_id = :collection_id "
                    "AND cmetadata->>'owner_id' = :owner_id "
                    "AND to_tsvector('simple', document) "
                    "    @@ plainto_tsquery('simple', :query) "
                    "ORDER BY ts_rank("
                    "  to_tsvector('simple', document), plainto_tsquery('simple', :query)"
                    ") DESC "
                    "LIMIT :top_k"
                ),
                {
                    "collection_id": collection_uuid,
                    "owner_id": owner_id,
                    "query": query,
                    "top_k": top_k,
                },
            )
            rows = result.fetchall()

        return [
            Document(
                id=str(metadata.get("doc_id") or metadata.get("filename") or "unknown"),
                content=content,
                metadata=metadata,
            )
            for content, metadata in rows
        ]

    async def _collection_uuid(self, connection: Any) -> object | None:
        result = await connection.execute(
            text("SELECT uuid FROM langchain_pg_collection WHERE name = :name LIMIT 1"),
            {"name": self.collection_name},
        )
        row = result.fetchone()
        return row[0] if row else None

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        # Attempt deletion through LangChain (only works when the chunk id == doc_id — rare,
        # because chunks have ids like "{doc_id}::{index}"). Treat it as best effort: the real
        # deletion is the metadata DELETE below. A database-level error is logged rather than
        # silently swallowed (`suppress(Exception)` used to hide everything).
        try:
            await self.vector_store.adelete(ids=[doc_id])
        except SQLAlchemyError:
            logger.warning(
                "LangChain adelete failed for doc_id=%s — falling back to metadata deletion",
                doc_id,
                exc_info=True,
            )

        # Exact deletion of all the document's chunks via metadata. The owner_id filter
        # protects against removing someone else's vectors on a name collision.
        async with db_connection() as connection:
            collection_uuid = await self._collection_uuid(connection)
            if collection_uuid is not None:
                # Match on the parent document id (namespaced per user) and, for safety,
                # on owner_id as well.
                await connection.execute(
                    text(
                        "DELETE FROM langchain_pg_embedding "
                        "WHERE collection_id = :collection_id "
                        "AND cmetadata->>'doc_id' = :doc_id "
                        "AND cmetadata->>'owner_id' = :owner_id"
                    ),
                    {
                        "collection_id": collection_uuid,
                        "doc_id": doc_id,
                        "owner_id": owner_id,
                    },
                )
