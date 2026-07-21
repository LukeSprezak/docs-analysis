from functools import lru_cache
from typing import cast

from app.shared.config import settings
from app.shared.enums import RerankerProvider

from ...domain.repositories import RerankerService
from .llm_factory import LLMFactory
from .reranker import (
    CohereReranker,
    CrossEncoderScorer,
    LLMReranker,
    LocalCrossEncoderReranker,
    NoOpReranker,
)


@lru_cache(maxsize=1)
def _load_bge_scorer() -> CrossEncoderScorer:
    # Wagi modelu ładujemy raz (drogie) i cache'ujemy. Import leniwy — pakiet
    # `sentence-transformers` (z torchem) jest potrzebny tylko dla tego wariantu.
    from sentence_transformers import CrossEncoder

    # cast: sentence-transformers nie ma stubów (CrossEncoder jest Any), a my znamy
    # potrzebny kontrakt (metoda predict).
    return cast(CrossEncoderScorer, CrossEncoder(settings.BGE_RERANKER_MODEL))


class RerankerFactory:
    @staticmethod
    def get_reranker() -> RerankerService:
        match settings.RERANKER_PROVIDER:
            case RerankerProvider.LLM:
                return LLMReranker(llm=LLMFactory.get_llm())
            case RerankerProvider.COHERE:
                if not settings.COHERE_API_KEY:
                    raise ValueError("RERANKER_PROVIDER=cohere wymaga ustawienia COHERE_API_KEY")
                # Import leniwy — pakiet `cohere` jest potrzebny tylko dla tego wariantu.
                import cohere

                client = cohere.ClientV2(settings.COHERE_API_KEY)
                # cohere zwraca własny bogaty typ odpowiedzi; nasz minimalny Protocol
                # (index/results) opisuje tylko to, czego używamy — stąd most typów.
                return CohereReranker(
                    client=client,  # type: ignore[arg-type]
                    model=settings.COHERE_RERANK_MODEL,
                )
            case RerankerProvider.BGE:
                return LocalCrossEncoderReranker(scorer=_load_bge_scorer())
            case _:
                return NoOpReranker()
