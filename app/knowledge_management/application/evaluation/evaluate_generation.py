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
    """Scores the quality of generated answers (LLM-as-judge, RAGAS style).

    For each question: retrieves the context with the same pipeline production uses,
    generates an answer via `RAGService`, and then the judge scores faithfulness (grounding
    in the context) and answer relevance (how well it addresses the question).
    """

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
            answer_relevance=await self.judge.score_answer_relevance(example.question, answer),
        )

    async def evaluate(
        self, examples: Sequence[EvaluationExample], owner_id: str
    ) -> tuple[GenerationMetrics, list[GenerationExampleResult]]:
        results = [await self.evaluate_example(example, owner_id) for example in examples]
        faithfulness = [r.faithfulness for r in results if r.faithfulness is not None]
        answer_relevance = [r.answer_relevance for r in results if r.answer_relevance is not None]
        metrics = GenerationMetrics(
            example_count=len(results),
            mean_faithfulness=retrieval_metrics.mean(faithfulness),
            mean_answer_relevance=retrieval_metrics.mean(answer_relevance),
            scored_faithfulness_count=len(faithfulness),
            scored_answer_relevance_count=len(answer_relevance),
        )
        return metrics, results
