"""Czyste, deterministyczne metryki jakości retrievalu.

Funkcje operują na listach identyfikatorów dokumentów (kolejność = ranking z retrievalu)
oraz na zbiorze identyfikatorów trafnych. Nie mają zależności od LLM ani sieci — to one
stanowią bramkę regresji nadającą się do CI.
"""

from collections.abc import Iterable, Sequence


def is_hit_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> bool:
    """Czy wśród pierwszych ``k`` zwróconych dokumentów jest jakikolwiek trafny."""
    relevant = set(relevant_document_ids)
    return any(document_id in relevant for document_id in retrieved_document_ids[:k])


def precision_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> float:
    """Odsetek trafnych wśród faktycznie zwróconych pozycji (do min(k, liczba zwróconych)).

    Dzielimy przez liczbę realnie zwróconych pozycji, a nie sztywno przez ``k`` — inaczej
    karalibyśmy retrieval za to, że trafnych dokumentów jest po prostu mniej niż ``k``.
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
    """Odsetek trafnych dokumentów pokrytych przez pierwsze ``k`` pozycji."""
    relevant = set(relevant_document_ids)
    if not relevant:
        return 0.0
    top = set(retrieved_document_ids[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
) -> float:
    """1/(pozycja pierwszego trafnego dokumentu); 0.0 gdy żaden nie trafił."""
    relevant = set(relevant_document_ids)
    for position, document_id in enumerate(retrieved_document_ids, start=1):
        if document_id in relevant:
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Średnia arytmetyczna; 0.0 dla pustej sekwencji (brak przykładów)."""
    return sum(values) / len(values) if values else 0.0