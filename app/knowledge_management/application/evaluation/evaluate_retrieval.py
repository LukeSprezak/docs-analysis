from collections.abc import Callable, Sequence

from ...domain.document_identity import citation_label
from ...domain.evaluation import (
    UNCATEGORIZED,
    EvaluationExample,
    RetrievalExampleResult,
    RetrievalMetrics,
)
from ...domain.models import Document
from . import retrieval_metrics
from .retrieval_pipeline import RetrievalPipeline


def default_document_identifier(document: Document) -> str:
    """The parent document identifier used to match relevance — the same human-readable name
    a citation shows, because that is how golden sets are written."""
    return citation_label(document)


class RetrievalEvaluator:
    """Computes deterministic retrieval metrics over a golden set of reference questions."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        document_identifier: Callable[[Document], str] = default_document_identifier,
    ) -> None:
        self.pipeline = pipeline
        self._document_identifier = document_identifier

    async def evaluate_example(
        self, example: EvaluationExample, owner_id: str
    ) -> RetrievalExampleResult:
        documents = await self.pipeline.retrieve(example.question, owner_id)
        retrieved_document_ids = [self._document_identifier(document) for document in documents]
        top_k = self.pipeline.top_k
        relevant = example.relevant_document_ids
        return RetrievalExampleResult(
            question=example.question,
            retrieved_document_ids=retrieved_document_ids,
            relevant_document_ids=relevant,
            is_hit=retrieval_metrics.is_hit_at_k(retrieved_document_ids, relevant, top_k),
            reciprocal_rank=retrieval_metrics.reciprocal_rank(retrieved_document_ids, relevant),
            precision_at_k=retrieval_metrics.precision_at_k(
                retrieved_document_ids, relevant, top_k
            ),
            recall_at_k=retrieval_metrics.recall_at_k(retrieved_document_ids, relevant, top_k),
            category=example.category,
        )

    async def evaluate(
        self, examples: Sequence[EvaluationExample], owner_id: str
    ) -> tuple[RetrievalMetrics, list[RetrievalExampleResult]]:
        results = [await self.evaluate_example(example, owner_id) for example in examples]
        return aggregate_metrics(results), results


def aggregate_metrics(results: Sequence[RetrievalExampleResult]) -> RetrievalMetrics:
    """Averages per-question results into the headline metrics.

    Extracted so the same aggregation can be applied to a subset — grouping by category is
    just this function over each group.
    """
    return RetrievalMetrics(
        example_count=len(results),
        hit_rate=retrieval_metrics.mean([1.0 if r.is_hit else 0.0 for r in results]),
        mean_reciprocal_rank=retrieval_metrics.mean([r.reciprocal_rank for r in results]),
        mean_precision_at_k=retrieval_metrics.mean([r.precision_at_k for r in results]),
        mean_recall_at_k=retrieval_metrics.mean([r.recall_at_k for r in results]),
    )


def group_by_category(
    results: Sequence[RetrievalExampleResult],
) -> dict[str, RetrievalMetrics]:
    """Metrics per question category, in first-appearance order."""
    grouped: dict[str, list[RetrievalExampleResult]] = {}
    for result in results:
        grouped.setdefault(result.category or UNCATEGORIZED, []).append(result)
    return {category: aggregate_metrics(group) for category, group in grouped.items()}
