"""Uruchamialny harness ewaluacji RAG (AI-9).

Metryki retrievalu liczą się zawsze (offline, deterministyczne). Metryki generacji
(LLM-as-judge) dokładają się tylko, gdy ustawiono ``EVAL_JUDGE_PROVIDER=llm``.

Uwaga: realny przebieg liczy jakość na PRAWDZIWYCH embeddingach/LLM — wymaga
skonfigurowanego providera i wgranych dokumentów dla danego ``owner_id``. To narzędzie
do uruchamiania na żądanie / w nightly, nie bezsieciowy test jednostkowy (metryki same
w sobie są testowane offline w `tests/`).

Przykład:
    uv run python -m app.knowledge_management.application.evaluation.run_evaluation \\
        --dataset eval/golden_set.json --owner-id <user_id>
"""

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
    """Składa pełny raport. Sekcja generacji pojawia się tylko, gdy podano evaluator."""
    retrieval, retrieval_details = await retrieval_evaluator.evaluate(examples, owner_id)
    report = EvaluationReport(retrieval=retrieval, retrieval_details=retrieval_details)
    if generation_evaluator is not None:
        generation, generation_details = await generation_evaluator.evaluate(examples, owner_id)
        report.generation = generation
        report.generation_details = generation_details
    return report


def format_report(report: EvaluationReport) -> str:
    """Czytelne podsumowanie raportu do konsoli."""
    retrieval = report.retrieval
    lines = [
        f"Ewaluacja na {retrieval.example_count} pytaniach",
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
            "GENERACJA (LLM-as-judge):",
            f"  faithfulness     : {generation.mean_faithfulness:.3f}",
            f"  answer_relevance : {generation.mean_answer_relevance:.3f}",
        ]
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval harness RAG (AI-9).")
    parser.add_argument(
        "--dataset", help="Ścieżka do golden setu JSON (domyślnie z EVAL_DATASET_PATH)."
    )
    parser.add_argument(
        "--owner-id",
        required=True,
        help="Właściciel dokumentów, na których liczymy retrieval (izolacja per user).",
    )
    parser.add_argument("--json", dest="json_output", help="Zapisz pełny raport do pliku JSON.")
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    # Importy ciężkich zależności w środku — sam moduł importuje się bez nich (np. w testach).
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
        # Zapis pliku to blokujące I/O — w wątku, żeby nie blokować event loopu (ASYNC230).
        await asyncio.to_thread(_write_json_report, arguments.json_output, report)


def _write_json_report(output_path: str, report: EvaluationReport) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(asdict(report), output_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
