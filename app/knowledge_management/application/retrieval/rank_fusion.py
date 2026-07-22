"""Reciprocal Rank Fusion (RRF) — combining several rankings into one.

Pure, deterministic and network-free (testable offline). Used by hybrid search to merge the
results of vector and full-text search (BM25/FTS).
"""

from collections.abc import Callable, Sequence

from ...domain.models import Document

# The smoothing constant from the original RRF paper. It dampens the weight of top
# positions, so a document relevant in both rankings beats one that is great in only one.
DEFAULT_RRF_SMOOTHING_CONSTANT = 60


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
