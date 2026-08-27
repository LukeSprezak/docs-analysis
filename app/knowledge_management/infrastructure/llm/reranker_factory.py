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
    # Model weights are loaded once (expensive) and cached. Lazy import — the
    # `sentence-transformers` package (with torch) is only needed for this variant.
    from sentence_transformers import CrossEncoder

    # cast: sentence-transformers ships no stubs (CrossEncoder is Any), while we do know
    # the contract we need (the predict method).
    return cast(CrossEncoderScorer, CrossEncoder(settings.BGE_RERANKER_MODEL))


class RerankerFactory:
    @staticmethod
    def get_reranker() -> RerankerService:
        match settings.RERANKER_PROVIDER:
            case RerankerProvider.LLM:
                return LLMReranker(llm=LLMFactory.get_llm())
            case RerankerProvider.COHERE:
                if not settings.COHERE_API_KEY:
                    raise ValueError("RERANKER_PROVIDER=cohere requires COHERE_API_KEY to be set")
                # Lazy import — the `cohere` package is only needed for this variant.
                import cohere

                client = cohere.ClientV2(settings.COHERE_API_KEY)
                # cohere returns its own rich response type; our minimal Protocol
                # (index/results) describes only what we use — hence the type bridge.
                return CohereReranker(
                    client=client,  # type: ignore[arg-type]
                    model=settings.COHERE_RERANK_MODEL,
                )
            case RerankerProvider.BGE:
                return LocalCrossEncoderReranker(scorer=_load_bge_scorer())
            case RerankerProvider.NONE:
                return NoOpReranker()
            case _:
                # No silent degradation: a typo in RERANKER_PROVIDER would otherwise start a
                # system that simply stops reranking, costing answer quality with nothing to
                # report it. Turning reranking off is a decision, so it has to be spelled
                # `none` — same rule as every other factory here.
                raise ValueError(
                    f"Unsupported Reranker provider: {settings.RERANKER_PROVIDER}"
                )
