import json
import re
from collections.abc import Sequence
from functools import partial
from typing import Protocol

import anyio
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ...domain.models import Document
from ...domain.repositories import RerankerService

RERANK_SYSTEM_PROMPT = """
Jesteś precyzyjnym systemem rerankingu. Oceniasz, które fragmenty najlepiej odpowiadają na
pytanie użytkownika. Zwracasz WYŁĄCZNIE tablicę JSON z indeksami fragmentów uszeregowanymi
od najbardziej do najmniej trafnego, np. [3, 0, 5]. Bez komentarza, bez markdownu.
"""


class NoOpReranker(RerankerService):
    """Brak rerankingu — zwraca pierwsze top_k (kolejność z vector search)."""

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        return documents[:top_k]


class CohereRerankResultItem(Protocol):
    index: int


class CohereRerankResponse(Protocol):
    results: Sequence[CohereRerankResultItem]


class CohereRerankClient(Protocol):
    """Minimalny kontrakt klienta Cohere potrzebny do rerankingu (ułatwia testy)."""

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> CohereRerankResponse: ...


class CohereReranker(RerankerService):
    """Reranking przez Cohere Rerank API (prawdziwy cross-encoder).

    Klient jest wstrzykiwany, dzięki czemu sama klasa nie importuje pakietu `cohere`
    (zależność pozostaje opcjonalna) i jest łatwo testowalna bez sieci.
    """

    def __init__(self, client: CohereRerankClient, model: str = "rerank-v3.5"):
        self._client = client
        self._model = model

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        if len(documents) <= 1:
            return documents[:top_k]

        # Klient Cohere jest synchroniczny — wołanie sieciowe w wątku, żeby nie blokować
        # event loopu.
        response = await anyio.to_thread.run_sync(
            partial(
                self._client.rerank,
                model=self._model,
                query=query,
                documents=[document.content for document in documents],
                top_n=min(top_k, len(documents)),
            )
        )
        return [documents[result.index] for result in response.results]


class CrossEncoderScorer(Protocol):
    """Kontrakt lokalnego cross-encodera (np. sentence-transformers CrossEncoder)."""

    def predict(self, sentence_pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class LocalCrossEncoderReranker(RerankerService):
    """Reranking lokalnym cross-encoderem (offline, bez API).

    Scorer jest wstrzykiwany, więc klasa nie importuje `sentence-transformers`
    (ciężka zależność z torchem pozostaje opcjonalna) i jest testowalna bez modelu.
    """

    def __init__(self, scorer: CrossEncoderScorer):
        self._scorer = scorer

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        if len(documents) <= 1:
            return documents[:top_k]

        pairs = [(query, document.content) for document in documents]
        # Predykcja cross-encodera jest CPU-bound (torch) — w wątku, by nie blokować pętli.
        scores = await anyio.to_thread.run_sync(self._scorer.predict, pairs)

        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [document for document, _ in ranked[:top_k]]


class LLMReranker(RerankerService):
    """Listwise reranking przez skonfigurowany LLM.

    Prosi model o uszeregowanie kandydatów wg trafności i zwraca najlepsze top_k.
    Przy nieparsowalnej odpowiedzi degraduje się łagodnie do kolejności wejściowej.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @staticmethod
    def _parse_ranked_indices(raw: str, document_count: int) -> list[int]:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []

        seen: set[int] = set()
        indices: list[int] = []
        for value in parsed:
            if isinstance(value, int) and 0 <= value < document_count and value not in seen:
                seen.add(value)
                indices.append(value)
        return indices

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        if len(documents) <= 1:
            return documents[:top_k]

        passages = "\n\n".join(
            f"[{index}] {document.content}" for index, document in enumerate(documents)
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RERANK_SYSTEM_PROMPT),
                ("human", "Pytanie:\n{query}\n\nFragmenty:\n{passages}"),
            ]
        )
        chain = prompt | self.llm
        response = await chain.ainvoke({"query": query, "passages": passages})

        ranked_indices = self._parse_ranked_indices(str(response.content), len(documents))
        if not ranked_indices:
            return documents[:top_k]

        reranked = [documents[index] for index in ranked_indices]
        # Dołóż fragmenty pominięte przez model (zachowując oryginalną kolejność),
        # żeby nie zgubić kandydatów przy niekompletnej odpowiedzi.
        ranked_set = set(ranked_indices)
        reranked.extend(
            document for index, document in enumerate(documents) if index not in ranked_set
        )
        return reranked[:top_k]