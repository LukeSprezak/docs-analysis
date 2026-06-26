from collections.abc import Sequence

from ...domain.evaluation import (
    EvaluationExample,
    GenerationExampleResult,
    GenerationMetrics,
)
from ...domain.repositories import AnswerJudge, RAGService
from . import retrieval_metrics
from .retrieval_pipeline import RetrievalPipeline


class GenerationEvaluator:
    """Evaluates the quality of the generated responses (LLM-as-judge, RAGAS-style)."""
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        rag_service: RAGService,
        judge: AnswerJudge,
    ) -> None:
        self.pipeline = pipeline
        self.rag_service = rag_service
        self.judge = judge

    async def evaluate_example(
        self, example: EvaluationExample, owner_id: str
    ) -> GenerationExampleResult:
        context = await self.pipeline.retrieve(example.question, owner_id)
        answer = await self.rag_service.answer_question(example.question, context)
        return GenerationExampleResult(
            question=example.question,
            answer=answer,
            faithfulness=await self.judge.score_faithfulness(answer, context),
            answer_relevance=await self.judge.score_answer_relevance(
                example.question, answer
            ),
        )

    async def evaluate(
        self, examples: Sequence[EvaluationExample], owner_id: str
    ) -> tuple[GenerationMetrics, list[GenerationExampleResult]]:
        results = [
            await self.evaluate_example(example, owner_id) for example in examples
        ]
        metrics = GenerationMetrics(
            example_count=len(results),
            mean_faithfulness=retrieval_metrics.mean([r.faithfulness for r in results]),
            mean_answer_relevance=retrieval_metrics.mean([r.answer_relevance for r in results]),
        )
        return metrics, results
