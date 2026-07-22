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

A/B of the retrieval pipeline with and without the knowledge graph (same corpus, same
reranker — the graph is the only variable). Copy `eval/golden_set.template.json` first; it
explains the question categories the report breaks the numbers down by:

    KNOWLEDGE_GRAPH_PROVIDER=neo4j uv run python -m \\
        app.knowledge_management.application.evaluation.run_evaluation \\
        --dataset eval/golden_set.json --owner-id <user_id> --compare-graph
"""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict

from ...domain.evaluation import EvaluationExample, EvaluationReport, RetrievalMetrics
from ...domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from ...domain.repositories import KnowledgeGraphRepo
from .dataset import load_examples
from .evaluate_generation import GenerationEvaluator
from .evaluate_retrieval import RetrievalEvaluator, group_by_category


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
    lines = [
        f"Retrieval comparison over {baseline.retrieval.example_count} questions",
        "",
        "OVERALL",
        *_comparison_rows(baseline.retrieval, candidate.retrieval, candidate_label),
    ]

    # The per-category view is the one to read. The overall average mixes question shapes the
    # graph affects in opposite directions, so it can sit near zero while both halves moved a
    # long way — see the category note in `domain/evaluation.py`.
    baseline_by_category = group_by_category(baseline.retrieval_details)
    candidate_by_category = group_by_category(candidate.retrieval_details)
    for category, baseline_metrics in baseline_by_category.items():
        candidate_metrics = candidate_by_category.get(category)
        if candidate_metrics is None:
            continue
        lines += [
            "",
            f"{category.upper()} ({baseline_metrics.example_count} questions)",
            *_comparison_rows(baseline_metrics, candidate_metrics, candidate_label),
        ]

    lines += ["", *_format_regressions(baseline, candidate)]
    return "\n".join(lines)


def find_regressions(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> dict[str, list[str]]:
    """Metrics that got worse, per question category — keyed by category, empty when clean.

    Scoped per category rather than over the whole set on purpose. A graph that wins big on
    cross-document questions while breaking single-passage ones nets out to a flat overall
    average, so an overall-only check reports "no regressions" for exactly the outcome you
    most need to see. This is the shape to gate CI on.
    """
    baseline_by_category = group_by_category(baseline.retrieval_details)
    candidate_by_category = group_by_category(candidate.retrieval_details)

    regressions: dict[str, list[str]] = {}
    for category, baseline_metrics in baseline_by_category.items():
        candidate_metrics = candidate_by_category.get(category)
        if candidate_metrics is None:
            continue
        regressed = [
            label
            for label, attribute in _COMPARISON_METRICS
            if getattr(candidate_metrics, attribute) < getattr(baseline_metrics, attribute)
        ]
        if regressed:
            regressions[category] = regressed
    return regressions


def _format_regressions(baseline: EvaluationReport, candidate: EvaluationReport) -> list[str]:
    regressions = find_regressions(baseline, candidate)
    if not regressions:
        return ["Regressions: none"]
    return [
        "Regressions (per category):",
        *(f"  {category}: {', '.join(metrics)}" for category, metrics in regressions.items()),
    ]


_COMPARISON_METRICS = [
    ("hit_rate@k", "hit_rate"),
    ("MRR", "mean_reciprocal_rank"),
    ("precision@k", "mean_precision_at_k"),
    ("recall@k", "mean_recall_at_k"),
]


def _comparison_rows(
    baseline: RetrievalMetrics, candidate: RetrievalMetrics, candidate_label: str
) -> list[str]:
    rows = [
        f"  {'metric':<16}{'vector-only':>13}{candidate_label:>15}{'delta':>10}",
        "  " + "-" * 54,
    ]
    for label, attribute in _COMPARISON_METRICS:
        before = getattr(baseline, attribute)
        after = getattr(candidate, attribute)
        rows.append(f"  {label:<16}{before:>13.3f}{after:>15.3f}{after - before:>+10.3f}")
    return rows


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
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "Exit non-zero when any metric got worse in any question category. Off by default"
            " so an exploratory run still prints its numbers instead of just failing."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.fail_on_regression and not arguments.compare_graph:
        # There is nothing to regress against without a second pipeline to compare to;
        # silently ignoring the flag would make a CI gate look active when it never runs.
        parser.error("--fail-on-regression requires --compare-graph")
    return arguments


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
        # The report is written before the gate fires: a failing run is exactly the one whose
        # per-question details you want to inspect.
        if arguments.json_output:
            await asyncio.to_thread(_write_json_report, arguments.json_output, candidate)
        if arguments.fail_on_regression:
            regressions = find_regressions(baseline, candidate)
            if regressions:
                raise SystemExit(
                    f"FAILED: retrieval regressed in {len(regressions)} "
                    f"categor{'y' if len(regressions) == 1 else 'ies'} "
                    f"({', '.join(regressions)}) — see the breakdown above."
                )
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
