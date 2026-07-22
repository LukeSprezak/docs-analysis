"""Gathers retrieval candidates from every source and merges them into one ranking.

Both the Q&A and the chat use case need the same thing: passages from the vector store plus,
when a knowledge graph is configured, the facts connected to whatever the question mentions.
Doing that here keeps the fusion in one place and keeps the use cases about their own job.

There is no `if graph_enabled` anywhere: with the graph off the repository is the null one,
its result is empty, and fusing a ranking with an empty one returns the first ranking in its
original order. "Graph disabled" and "graph found nothing" are then the same code path — and
the second case has to work correctly regardless.
"""

from ...domain.models import Document
from ...domain.repositories import KnowledgeGraphRepo, VectorStoreRepo
from .rank_fusion import fuse_documents


def retrieval_key(document: Document) -> str:
    """Identity of a candidate across rankings, for deduplication during fusion.

    Vector hits are chunks and identify as `doc_id::chunk_index`. Graph hits are statements
    built from a triple, with no chunk to point at, so they fall back to their text — two
    identical statements are the same fact regardless of which entity lookup surfaced them.
    """
    doc_id = document.metadata.get("doc_id")
    chunk_index = document.metadata.get("chunk_index")
    if doc_id is not None and chunk_index is not None:
        return f"{doc_id}::{chunk_index}"
    return f"{document.id}::{document.content}"


class CandidateRetriever:
    def __init__(self, vector_repo: VectorStoreRepo, graph_repo: KnowledgeGraphRepo):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo

    async def retrieve(self, query: str, owner_id: str, candidate_count: int) -> list[Document]:
        """Candidates for `query`, best first, limited to what `owner_id` owns."""
        passages = await self.vector_repo.search(query, owner_id, top_k=candidate_count)
        facts = await self.graph_repo.search_related(query, owner_id, top_k=candidate_count)
        return fuse_documents([passages, facts], top_k=candidate_count, key_of=retrieval_key)
