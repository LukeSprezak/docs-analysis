from collections.abc import Iterable, Sequence


def is_hit_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> bool:
    """Is there any match among the first ``k`` documents returned?"""

    relevant = set(relevant_document_ids)
    return any(document_id in relevant for document_id in retrieved_document_ids[:k])


def precision_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> float:
    """Percentage of hits among the items actually returned (out of min(k, number of items returned))"""

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
    """Percentage of relevant documents covered by the first ``k`` items."""
    relevant = set(relevant_document_ids)
    if not relevant:
        return 0.0
    top = set(retrieved_document_ids[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
) -> float:
    """1/(position of the first matching document); 0.0 if none matched."""
    relevant = set(relevant_document_ids)
    for position, document_id in enumerate(retrieved_document_ids, start=1):
        if document_id in relevant:
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence (no examples)."""
    return sum(values) / len(values) if values else 0.0
