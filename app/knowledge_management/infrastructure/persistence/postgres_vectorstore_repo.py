import logging
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.shared.database import db_connection, get_engine

from ...domain.models import Document
from ...domain.repositories import VectorStoreRepo
from ..text.text_chunker import TextChunker
from .rank_fusion import fuse_documents

logger = logging.getLogger(__name__)


def _chunk_key(document: Document) -> str:
    return f"{document.metadata.get('doc_id')}::{document.metadata.get('chunk_index')}"


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

        for document_id in {document.id for document in owned_documents}:
            await self.delete_by_document_id(document_id, owner_id)

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

    async def _vector_search(
        self, query: str, owner_id: str, top_k: int
    ) -> list[Document]:
        results = await self.vector_store.asimilarity_search(
            query, k=top_k, filter={"owner_id": {"$eq": owner_id}}
        )
        return [
            Document(
                id=str(res.metadata.get("doc_id") or res.metadata.get("filename") or "unknown"),
                content=res.page_content,
                metadata=res.metadata,
            )
            for res in results
        ]

    async def _hybrid_search(
        self, query: str, owner_id: str, top_k: int
    ) -> list[Document]:
        vector_documents = await self._vector_search(query, owner_id, top_k)
        keyword_documents = await self._keyword_search(query, owner_id, top_k)
        return fuse_documents(
            [vector_documents, keyword_documents], top_k=top_k, key_of=_chunk_key
        )

    async def _keyword_search(
        self, query: str, owner_id: str, top_k: int
    ) -> list[Document]:
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
        try:
            await self.vector_store.adelete(ids=[doc_id])
        except SQLAlchemyError:
            logger.warning(
                "LangChain adelete nie powiodło się dla doc_id=%s — "
                "kontynuuję usuwaniem po metadanych",
                doc_id,
                exc_info=True,
            )

        async with db_connection() as connection:
            collection_uuid = await self._collection_uuid(connection)
            if collection_uuid is not None:
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
