"""Pure, deterministic retrieval quality metrics.

The functions operate on lists of document identifiers (order = the ranking produced by
retrieval) and on the set of relevant identifiers. They depend on neither an LLM nor the
network — which is what makes them a regression gate suitable for CI.
"""

from collections.abc import Iterable, Sequence


def is_hit_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> bool:
    """Whether any of the first ``k`` returned documents is relevant."""
    relevant = set(relevant_document_ids)
    return any(document_id in relevant for document_id in retrieved_document_ids[:k])


def precision_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> float:
    """Share of relevant items among those actually returned (up to min(k, returned count)).

    We divide by the number of items actually returned rather than by ``k`` — otherwise we
    would penalise retrieval simply because fewer than ``k`` relevant documents exist.
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant_document_ids)
    top = retrieved_document_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for document_id in top if document_id in relevant)
    return hits / len(top)


def recall_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> float:
    """Share of the relevant documents covered by the first ``k`` positions."""
    relevant = set(relevant_document_ids)
    if not relevant:
        return 0.0
    top = set(retrieved_document_ids[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
) -> float:
    """1/(position of the first relevant document); 0.0 when none matched."""
    relevant = set(relevant_document_ids)
    for position, document_id in enumerate(retrieved_document_ids, start=1):
        if document_id in relevant:
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence (no examples)."""
    return sum(values) / len(values) if values else 0.0
