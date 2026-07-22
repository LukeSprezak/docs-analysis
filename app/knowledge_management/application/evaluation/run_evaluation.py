"""Runnable RAG evaluation harness (AI-9).

Retrieval metrics are always computed (offline, deterministic). Generation metrics
(LLM-as-judge) are added only when ``EVAL_JUDGE_PROVIDER=llm`` is set.

Note: a real run measures quality against REAL embeddings/LLM — it needs a configured
provider and documents uploaded for the given ``owner_id``. This is a tool to run on demand
or nightly, not an offline unit test (the metrics themselves are tested offline in
`tests/`).

Example:
    uv run python -m app.knowledge_management.application.evaluation.run_evaluation \\
        --dataset eval/golden_set.json --owner-id <user_id>
"""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict

from ...domain.evaluation import EvaluationExample, EvaluationReport
from ...domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from ...domain.repositories import KnowledgeGraphRepo
from .dataset import load_examples
from .evaluate_generation import GenerationEvaluator
from .evaluate_retrieval import RetrievalEvaluator


async def build_report(
    examples: Sequence[EvaluationExample],
    owner_id: str,
    retrieval_evaluator: RetrievalEvaluator,
    generation_evaluator: GenerationEvaluator | None = None,
) -> EvaluationReport:
    """Assembles the full report. The generation section appears only when an evaluator is given."""
    retrieval, retrieval_details = await retrieval_evaluator.evaluate(examples, owner_id)
    report = EvaluationReport(retrieval=retrieval, retrieval_details=retrieval_details)
    if generation_evaluator is not None:
        generation, generation_details = await generation_evaluator.evaluate(examples, owner_id)
        report.generation = generation
        report.generation_details = generation_details
    return report


def format_report(report: EvaluationReport) -> str:
    """A human-readable summary of the report for the console."""
    retrieval = report.retrieval
    lines = [
        f"Evaluation over {retrieval.example_count} questions",
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


def format_comparison(
    baseline: EvaluationReport, candidate: EvaluationReport, candidate_label: str
) -> str:
    """Side-by-side retrieval metrics with the delta, for an A/B of two pipelines.

    Only retrieval metrics are compared: they are deterministic, so a difference is caused by
    the pipeline change rather than by LLM sampling noise. A judge-scored generation delta
    would need repeated runs to separate signal from variance.
    """
    rows = [
        ("hit_rate@k", "hit_rate"),
        ("MRR", "mean_reciprocal_rank"),
        ("precision@k", "mean_precision_at_k"),
        ("recall@k", "mean_recall_at_k"),
    ]
    lines = [
        f"Retrieval comparison over {baseline.retrieval.example_count} questions",
        "",
        f"{'metric':<18}{'vector-only':>13}{candidate_label:>15}{'delta':>10}",
        "-" * 56,
    ]
    for label, attribute in rows:
        before = getattr(baseline.retrieval, attribute)
        after = getattr(candidate.retrieval, attribute)
        lines.append(f"{label:<18}{before:>13.3f}{after:>15.3f}{after - before:>+10.3f}")

    regressed = [
        label
        for label, attribute in rows
        if getattr(candidate.retrieval, attribute) < getattr(baseline.retrieval, attribute)
    ]
    lines += ["", f"Regressions: {', '.join(regressed) if regressed else 'none'}"]
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG eval harness (AI-9).")
    parser.add_argument(
        "--dataset", help="Path to the JSON golden set (defaults to EVAL_DATASET_PATH)."
    )
    parser.add_argument(
        "--owner-id",
        required=True,
        help="Owner of the documents retrieval is measured on (per-user isolation).",
    )
    parser.add_argument("--json", dest="json_output", help="Write the full report to a JSON file.")
    parser.add_argument(
        "--compare-graph",
        action="store_true",
        help=(
            "Run retrieval twice over the same corpus — vector-only and vector+knowledge-graph"
            " — and print the metric delta. Requires KNOWLEDGE_GRAPH_PROVIDER to be set to a"
            " real graph, and that corpus to have been uploaded with the graph enabled."
        ),
    )
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    # Heavy dependencies are imported inside — the module itself imports without them (e.g. in tests).
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

    def build_pipeline(graph_repo: KnowledgeGraphRepo | None = None) -> RetrievalPipeline:
        return RetrievalPipeline(
            vector_repo=get_vector_repo(),
            reranker=RerankerFactory.get_reranker(),
            candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
            top_k=settings.RETRIEVAL_TOP_K,
            graph_repo=graph_repo,
        )

    if arguments.compare_graph:
        from app.shared.dependencies import get_graph_repo

        graph_repo = get_graph_repo()
        if isinstance(graph_repo, NullKnowledgeGraphRepo):
            raise SystemExit(
                "--compare-graph needs a real graph: set KNOWLEDGE_GRAPH_PROVIDER=neo4j. "
                "With the null graph both sides of the comparison would be identical."
            )
        # Same corpus, same reranker, same k — the only difference is the extra candidate source.
        baseline = await build_report(
            examples, arguments.owner_id, RetrievalEvaluator(build_pipeline())
        )
        candidate = await build_report(
            examples, arguments.owner_id, RetrievalEvaluator(build_pipeline(graph_repo))
        )
        print(format_comparison(baseline, candidate, candidate_label="vector+graph"))
        if arguments.json_output:
            await asyncio.to_thread(_write_json_report, arguments.json_output, candidate)
        return

    pipeline = build_pipeline()
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
        # Writing the file is blocking I/O — run it in a thread so the event loop stays free (ASYNC230).
        await asyncio.to_thread(_write_json_report, arguments.json_output, report)


def _write_json_report(output_path: str, report: EvaluationReport) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(asdict(report), output_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
