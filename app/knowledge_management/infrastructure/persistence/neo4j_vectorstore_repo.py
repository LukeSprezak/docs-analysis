"""Vector store backed by Neo4j's native vector index.

Structurally the twin of `postgres_vectorstore_repo`: chunk before embedding, stamp the
owner onto every chunk, and combine vector with keyword hits via RRF when hybrid retrieval
is on. Chunks live as `(:Chunk {id, text, embedding, owner_id, doc_id, chunk_index, …})`
nodes — the metadata becomes node properties, which is what lets the Cypher below filter on
`owner_id` directly.

Two deliberate departures from the LangChain defaults:

* **Hybrid search is assembled here, not delegated.** `Neo4jVector` supports
  `search_type=HYBRID`, but it *rejects metadata filters in that mode*
  ("Filters are not supported with hybrid search"). Owner isolation is not negotiable, so
  hybrid runs as two filtered queries fused with the shared `fuse_documents` — the same
  shape as the Postgres adapter.
* **Indexes are created explicitly.** `Neo4jVector.__init__` connects and verifies the
  server but does *not* create the vector index (only the `from_*` constructors do), so
  `_ensure_indexes` does it and is idempotent.

`Neo4jVector`'s async methods are currently `run_in_executor` wrappers around the sync
driver rather than native async. They still keep the event loop free, which is what matters
here; the raw Cypher below goes through the same helper for consistency.
"""

import logging
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_neo4j import Neo4jVector

from ...application.retrieval.rank_fusion import fuse_documents
from ...domain.models import Document
from ...domain.repositories import VectorStoreRepo
from ..text.text_chunker import TextChunker

logger = logging.getLogger(__name__)


def _chunk_key(document: Document) -> str:
    """Chunk identity across rankings (vector vs keyword) for the RRF fusion."""
    return f"{document.metadata.get('doc_id')}::{document.metadata.get('chunk_index')}"


class Neo4jVectorStoreRepo(VectorStoreRepo):
    def __init__(
        self,
        embeddings: Embeddings,
        url: str,
        username: str,
        password: str,
        database: str | None = None,
        index_name: str = "document_chunks",
        keyword_index_name: str = "document_chunks_keyword",
        node_label: str = "Chunk",
        chunker: TextChunker | None = None,
        enable_hybrid_search: bool = False,
    ) -> None:
        self.embeddings = embeddings
        self.chunker = chunker or TextChunker()
        self.enable_hybrid_search = enable_hybrid_search
        self.node_label = node_label

        self.vector_store = Neo4jVector(
            embedding=embeddings,
            url=url,
            username=username,
            password=password,
            database=database,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            node_label=node_label,
        )
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Creates the vector index (and the full-text one for hybrid) if absent."""
        if self.vector_store.retrieve_existing_index() is None:
            self.vector_store.create_new_index()
        if self.enable_hybrid_search:
            text_property = self.vector_store.text_node_property
            if self.vector_store.retrieve_existing_fts_index([text_property]) is None:
                self.vector_store.create_new_keyword_index([text_property])

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        # Stamp the owner onto every document BEFORE chunking — chunks inherit the metadata,
        # so owner_id lands on every node and lets retrieval be filtered per user.
        owned_documents = [
            Document(
                id=document.id,
                content=document.content,
                metadata={**document.metadata, "owner_id": owner_id},
            )
            for document in documents
        ]

        # Re-upload: drop the document's existing chunks BEFORE inserting the new ones, or a
        # shorter new version leaves the old high-index chunks orphaned in the graph.
        for document_id in {document.id for document in owned_documents}:
            await self.delete_by_document_id(document_id, owner_id)

        chunks = self.chunker.chunk_many(owned_documents)
        if not chunks:
            return

        lc_docs = [
            LCDocument(page_content=chunk.content, metadata=chunk.metadata or {})
            for chunk in chunks
        ]
        await self.vector_store.aadd_documents(lc_docs, ids=[chunk.id for chunk in chunks])

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        if self.enable_hybrid_search:
            return await self._hybrid_search(query, owner_id, top_k)
        return await self._vector_search(query, owner_id, top_k)

    async def _vector_search(self, query: str, owner_id: str, top_k: int) -> list[Document]:
        results = await self.vector_store.asimilarity_search(
            query, k=top_k, filter={"owner_id": {"$eq": owner_id}}
        )
        return [self._to_document(res.page_content, res.metadata) for res in results]

    async def _hybrid_search(self, query: str, owner_id: str, top_k: int) -> list[Document]:
        """Combines vector and full-text hits (Neo4j full-text index) via RRF."""
        vector_documents = await self._vector_search(query, owner_id, top_k)
        keyword_documents = await self._keyword_search(query, owner_id, top_k)
        return fuse_documents([vector_documents, keyword_documents], top_k=top_k, key_of=_chunk_key)

    async def _keyword_search(self, query: str, owner_id: str, top_k: int) -> list[Document]:
        """Full-text search over chunk text, ranked by Lucene score and scoped to the owner.

        The owner filter is inside the query rather than applied to the results, so another
        user's chunks never leave the database. `queryNodes` is asked for extra candidates
        because the owner filter runs after the index lookup.
        """
        rows = await self._query(
            "CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $candidates}) "
            "YIELD node, score "
            "WHERE node.owner_id = $owner_id "
            "RETURN node.`" + self.vector_store.text_node_property + "` AS text, "
            "       node {.*, `" + self.vector_store.text_node_property + "`: Null, "
            "             `" + self.vector_store.embedding_node_property + "`: Null} AS metadata "
            "ORDER BY score DESC LIMIT $top_k",
            {
                "index_name": self.vector_store.keyword_index_name,
                "query": _escape_lucene(query),
                "owner_id": owner_id,
                "candidates": max(top_k * 5, 50),
                "top_k": top_k,
            },
        )
        return [self._to_document(row["text"], row["metadata"]) for row in rows]

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        # Both properties are matched: doc_id alone is already namespaced per user, and
        # pinning owner_id as well keeps the guarantee local to this query.
        await self._query(
            f"MATCH (chunk:`{self.node_label}`) "
            "WHERE chunk.doc_id = $doc_id AND chunk.owner_id = $owner_id "
            "DETACH DELETE chunk",
            {"doc_id": doc_id, "owner_id": owner_id},
        )

    async def close(self) -> None:
        """Closes the Bolt driver and its connection pool.

        `Neo4jVector` opens a driver in its constructor but exposes no way to shut it down,
        so we reach for the one it holds. Without this the pool is only reclaimed when the
        object is garbage collected, and the driver complains from `__del__` about being
        left open — the graph counterpart of `dispose_engine()` for the SQLAlchemy pool.
        """
        await run_in_executor(None, self.vector_store._driver.close)

    async def _query(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Runs Cypher off the event loop (the driver underneath is synchronous)."""
        return await run_in_executor(None, lambda: self.vector_store.query(cypher, params=params))

    @staticmethod
    def _to_document(content: str, metadata: dict[str, Any]) -> Document:
        return Document(
            # The parent document id from the chunk metadata — the node's own `id` is the
            # chunk id ("{doc_id}::{index}"), which is not what callers cite.
            id=str(metadata.get("doc_id") or metadata.get("filename") or "unknown"),
            content=content,
            metadata=metadata,
        )


def _escape_lucene(query: str) -> str:
    """Neutralizes Lucene syntax in user input.

    `db.index.fulltext.queryNodes` takes a Lucene query, so characters like `:` or `~` from a
    user's question would otherwise be parsed as operators — at best skewing results, at
    worst raising a syntax error mid-request.
    """
    special_characters = r'+-&|!(){}[]^"~*?:\/'
    return "".join(f"\\{char}" if char in special_characters else char for char in query)
