"""Reciprocal Rank Fusion (RRF) — łączenie kilku rankingów w jeden.

Czyste, deterministyczne i bezsieciowe (testowalne offline). Używane przez hybrid search
do złączenia wyników wyszukiwania wektorowego i pełnotekstowego (BM25/FTS).
"""

from collections.abc import Callable, Sequence

from ...domain.models import Document

# Stała wygładzająca z oryginalnej pracy o RRF. Tłumi wpływ wysokich pozycji, dzięki
# czemu dokument trafny w obu rankingach bije dokument świetny tylko w jednym.
DEFAULT_RRF_SMOOTHING_CONSTANT = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    smoothing_constant: int = DEFAULT_RRF_SMOOTHING_CONSTANT,
) -> list[str]:
    """Łączy rankingi (listy kluczy w kolejności trafności) w jeden ranking kluczy.

    Wynik RRF każdego klucza to suma 1/(smoothing_constant + pozycja) po wszystkich
    rankingach, w których wystąpił. Pozycje liczone od 0. Zwraca klucze posortowane
    malejąco po wyniku; remisy zachowują kolejność pierwszego wystąpienia (stabilnie).
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
    """Łączy kilka list dokumentów przez RRF i zwraca najlepsze ``top_k`` (zdeduplikowane).

    ``key_of`` wyznacza tożsamość dokumentu między listami (np. id fragmentu), żeby ten sam
    fragment z dwóch rankingów zliczył się jako jeden, a nie zdublował.
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
