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
    from sentence_transformers import CrossEncoder

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
                import cohere

                client = cohere.ClientV2(settings.COHERE_API_KEY)
                return CohereReranker(
                    client=client,  # type: ignore[arg-type]
                    model=settings.COHERE_RERANK_MODEL,
                )
            case RerankerProvider.BGE:
                return LocalCrossEncoderReranker(scorer=_load_bge_scorer())
            case _:
                return NoOpReranker()
