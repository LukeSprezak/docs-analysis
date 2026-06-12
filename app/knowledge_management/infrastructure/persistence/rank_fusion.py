from collections.abc import Callable, Sequence

from ...domain.models import Document

DEFAULT_RRF_SMOOTHING_CONSTANT = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    smoothing_constant: int = DEFAULT_RRF_SMOOTHING_CONSTANT,
) -> list[str]:
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
