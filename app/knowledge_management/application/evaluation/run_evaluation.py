import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict

from ...domain.evaluation import EvaluationExample, EvaluationReport
from .dataset import load_examples
from .evaluate_generation import GenerationEvaluator
from .evaluate_retrieval import RetrievalEvaluator


async def build_report(
    examples: Sequence[EvaluationExample],
    owner_id: str,
    retrieval_evaluator: RetrievalEvaluator,
    generation_evaluator: GenerationEvaluator | None = None,
) -> EvaluationReport:
    """It generates a full report. The generation section appears only if an evaluator is specified."""
    retrieval, retrieval_details = await retrieval_evaluator.evaluate(examples, owner_id)
    report = EvaluationReport(retrieval=retrieval, retrieval_details=retrieval_details)
    if generation_evaluator is not None:
        generation, generation_details = await generation_evaluator.evaluate(
            examples, owner_id
        )
        report.generation = generation
        report.generation_details = generation_details
    return report


def format_report(report: EvaluationReport) -> str:
    """A clear summary of the report for the console."""
    retrieval = report.retrieval
    lines = [
        f"Evaluation of {retrieval.example_count} questions",
        "",
        "RETRIEVAL:",
        f"  hit_rate@k       : {retrieval.hit_rate:.3f}",
        f"  MRR              : {retrieval.mean_reciprocal_rank:.3f}",
        f"  precision@k      : {retrieval.mean_precision_at_k:.3f}",
        f"  recall@k         : {retrieval.mean_recall_at_k:.3f}",
    ]
    if report.generation is not None:
        generation = report.generation
        lines += [
            "",
            "GENERATION (LLM-as-judge):",
            f"  faithfulness     : {generation.mean_faithfulness:.3f}",
            f"  answer_relevance : {generation.mean_answer_relevance:.3f}",
        ]
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval harness RAG (AI-9).")
    parser.add_argument(
        "--dataset", help="Path do golden setu JSON (default from EVAL_DATASET_PATH)."
    )
    parser.add_argument(
        "--owner-id",
        required=True,
        help="The owner of the documents we rely on retrieval (insulation per user).",
    )
    parser.add_argument("--json", dest="json_output", help="Save the full report to a file JSON.")
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    from app.knowledge_management.infrastructure.llm.answer_judge_factory import (
        AnswerJudgeFactory,
    )
    from app.knowledge_management.infrastructure.llm.langchain_rag_service import (
        LangChainRAGService,
    )
    from app.knowledge_management.infrastructure.llm.llm_factory import LLMFactory
    from app.knowledge_management.infrastructure.llm.reranker_factory import RerankerFactory
    from app.shared.config import settings
    from app.shared.dependencies import get_vector_repo

    from .retrieval_pipeline import RetrievalPipeline

    arguments = _parse_arguments(argv)
    examples = load_examples(arguments.dataset or settings.EVAL_DATASET_PATH)

    pipeline = RetrievalPipeline(
        vector_repo=get_vector_repo(),
        reranker=RerankerFactory.get_reranker(),
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
    )
    retrieval_evaluator = RetrievalEvaluator(pipeline)

    judge = AnswerJudgeFactory.get_judge()
    generation_evaluator = (
        GenerationEvaluator(pipeline, LangChainRAGService(llm=LLMFactory.get_llm()), judge)
        if judge is not None
        else None
    )

    report = await build_report(
        examples, arguments.owner_id, retrieval_evaluator, generation_evaluator
    )
    print(format_report(report))

    if arguments.json_output:
        await asyncio.to_thread(_write_json_report, arguments.json_output, report)


def _write_json_report(output_path: str, report: EvaluationReport) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(asdict(report), output_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
