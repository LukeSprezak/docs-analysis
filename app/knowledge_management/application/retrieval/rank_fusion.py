"""Reciprocal Rank Fusion (RRF) — combining several rankings into one.

Pure, deterministic and network-free (testable offline). Used by hybrid search to merge the
results of vector and full-text search (BM25/FTS), and by candidate retrieval to merge those
passages with the facts from the knowledge graph.

`retrieval_key` lives here too: fusion is the only reason document identity has to be pinned
down, and one definition shared by every caller is what keeps two rankings from disagreeing
about what "the same document" means.
"""

from collections.abc import Callable, Sequence

from ...domain.document_identity import chunk_id
from ...domain.models import Document

# The smoothing constant from the original RRF paper. It dampens the weight of top
# positions, so a document relevant in both rankings beats one that is great in only one.
DEFAULT_RRF_SMOOTHING_CONSTANT = 60


def retrieval_key(document: Document) -> str:
    """Identity of a candidate across rankings, for deduplication during fusion.

    Vector and keyword hits are chunks and identify as `doc_id#chunk_index`. Graph hits are
    statements built from a triple, with no chunk to point at, so they fall back to their text
    — two identical statements are the same fact regardless of which entity lookup surfaced
    them. The fallback is what keeps every metadata-less document from collapsing onto one
    shared key and silently deduplicating unrelated results down to a single hit.
    """
    doc_id = document.metadata.get("doc_id")
    chunk_index = document.metadata.get("chunk_index")
    if doc_id is not None and chunk_index is not None:
        return chunk_id(str(doc_id), chunk_index)
    return f"{document.id}::{document.content}"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    smoothing_constant: int = DEFAULT_RRF_SMOOTHING_CONSTANT,
) -> list[str]:
    """Merges rankings (lists of keys ordered by relevance) into a single ranking of keys.

    Each key's RRF score is the sum of 1/(smoothing_constant + position) across every
    ranking it appears in. Positions are 0-based. Returns the keys sorted by descending
    score; ties keep the order of first appearance (a stable sort).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (smoothing_constant + position + 1)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


def fuse_documents(
    document_lists: Sequence[Sequence[Document]],
    top_k: int,
    key_of: Callable[[Document], str],
    smoothing_constant: int = DEFAULT_RRF_SMOOTHING_CONSTANT,
) -> list[Document]:
    """Merges several document lists via RRF and returns the best ``top_k`` (deduplicated).

    ``key_of`` establishes document identity across the lists (e.g. the chunk id), so the
    same chunk appearing in two rankings counts once instead of being duplicated.
    """
    documents_by_key: dict[str, Document] = {}
    rankings: list[list[str]] = []
    for documents in document_lists:
        ranking: list[str] = []
        for document in documents:
            key = key_of(document)
            documents_by_key.setdefault(key, document)
            ranking.append(key)
        rankings.append(ranking)

    fused_keys = reciprocal_rank_fusion(rankings, smoothing_constant=smoothing_constant)
    return [documents_by_key[key] for key in fused_keys[:top_k]]
