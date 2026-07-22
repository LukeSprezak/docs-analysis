"""Per-category reporting for the golden set.

The split only earns its keep if the harness reports on the categories separately — an
overall average over question shapes the graph affects in opposite directions can sit at
zero while both halves moved a long way.
"""

import pytest

from app.knowledge_management.application.evaluation.dataset import load_examples, parse_examples
from app.knowledge_management.application.evaluation.evaluate_retrieval import group_by_category
from app.knowledge_management.application.evaluation.run_evaluation import (
    _parse_arguments,
    find_regressions,
    format_comparison,
)
from app.knowledge_management.domain.evaluation import (
    CATEGORY_CROSS_DOCUMENT,
    CATEGORY_SINGLE_PASSAGE,
    EvaluationReport,
    RetrievalExampleResult,
    RetrievalMetrics,
)

TEMPLATE_PATH = "eval/golden_set.template.json"


def _result(category: str | None, *, hit: bool) -> RetrievalExampleResult:
    return RetrievalExampleResult(
        question="q",
        retrieved_document_ids=["a.pdf"],
        relevant_document_ids=["a.pdf"],
        is_hit=hit,
        reciprocal_rank=1.0 if hit else 0.0,
        precision_at_k=1.0 if hit else 0.0,
        recall_at_k=1.0 if hit else 0.0,
        category=category,
    )


def _report(results: list[RetrievalExampleResult]) -> EvaluationReport:
    hit_rate = sum(1.0 for r in results if r.is_hit) / len(results)
    return EvaluationReport(
        retrieval=RetrievalMetrics(
            example_count=len(results),
            hit_rate=hit_rate,
            mean_reciprocal_rank=hit_rate,
            mean_precision_at_k=hit_rate,
            mean_recall_at_k=hit_rate,
        ),
        retrieval_details=results,
    )


def test_dataset_parses_the_category_field():
    examples = parse_examples(
        [
            {
                "question": "q",
                "relevant_document_ids": ["a.pdf"],
                "category": CATEGORY_CROSS_DOCUMENT,
            }
        ]
    )
    assert examples[0].category == CATEGORY_CROSS_DOCUMENT


def test_dataset_leaves_category_unset_when_absent():
    examples = parse_examples([{"question": "q", "relevant_document_ids": ["a.pdf"]}])
    assert examples[0].category is None


def test_group_by_category_aggregates_each_group_separately():
    grouped = group_by_category(
        [
            _result(CATEGORY_SINGLE_PASSAGE, hit=True),
            _result(CATEGORY_SINGLE_PASSAGE, hit=True),
            _result(CATEGORY_CROSS_DOCUMENT, hit=False),
        ]
    )

    assert grouped[CATEGORY_SINGLE_PASSAGE].hit_rate == 1.0
    assert grouped[CATEGORY_SINGLE_PASSAGE].example_count == 2
    assert grouped[CATEGORY_CROSS_DOCUMENT].hit_rate == 0.0


def test_uncategorized_questions_get_their_own_group():
    grouped = group_by_category([_result(None, hit=True)])
    assert "uncategorized" in grouped


def test_comparison_surfaces_opposite_movements_the_overall_average_hides():
    # The exact case the split exists for: the graph loses every single_passage question and
    # wins every cross_document one. The overall hit_rate is unchanged at 0.5, so a report
    # without the breakdown would read as "the graph changed nothing".
    baseline = _report(
        [
            _result(CATEGORY_SINGLE_PASSAGE, hit=True),
            _result(CATEGORY_CROSS_DOCUMENT, hit=False),
        ]
    )
    candidate = _report(
        [
            _result(CATEGORY_SINGLE_PASSAGE, hit=False),
            _result(CATEGORY_CROSS_DOCUMENT, hit=True),
        ]
    )
    assert baseline.retrieval.hit_rate == candidate.retrieval.hit_rate

    output = format_comparison(baseline, candidate, candidate_label="vector+graph")

    assert "SINGLE_PASSAGE" in output
    assert "CROSS_DOCUMENT" in output
    assert "-1.000" in output  # the regression on single_passage is visible
    assert "+1.000" in output  # so is the gain on cross_document


def test_regressions_are_detected_per_category_when_the_overall_average_is_flat():
    # The gating case: single_passage breaks, cross_document improves, overall is unchanged.
    # An overall-only check reports "no regressions" for precisely the outcome worth catching.
    baseline = _report(
        [
            _result(CATEGORY_SINGLE_PASSAGE, hit=True),
            _result(CATEGORY_CROSS_DOCUMENT, hit=False),
        ]
    )
    candidate = _report(
        [
            _result(CATEGORY_SINGLE_PASSAGE, hit=False),
            _result(CATEGORY_CROSS_DOCUMENT, hit=True),
        ]
    )
    assert baseline.retrieval.hit_rate == candidate.retrieval.hit_rate  # overall: no movement

    regressions = find_regressions(baseline, candidate)

    assert set(regressions) == {CATEGORY_SINGLE_PASSAGE}
    assert "hit_rate@k" in regressions[CATEGORY_SINGLE_PASSAGE]
    assert CATEGORY_SINGLE_PASSAGE in format_comparison(baseline, candidate, "vector+graph")


def test_no_regressions_when_every_category_improves():
    baseline = _report([_result(CATEGORY_SINGLE_PASSAGE, hit=False)])
    candidate = _report([_result(CATEGORY_SINGLE_PASSAGE, hit=True)])

    assert find_regressions(baseline, candidate) == {}
    assert "Regressions: none" in format_comparison(baseline, candidate, "vector+graph")


def test_fail_on_regression_is_rejected_without_a_comparison_to_run():
    # Accepting it silently would make a CI gate look active while never actually evaluating
    # anything — the worst failure mode for a guard.
    with pytest.raises(SystemExit):
        _parse_arguments(["--owner-id", "u1", "--fail-on-regression"])


def test_fail_on_regression_is_off_by_default():
    arguments = _parse_arguments(["--owner-id", "u1", "--compare-graph"])
    assert arguments.fail_on_regression is False
    assert arguments.compare_graph is True


def test_template_golden_set_is_valid_and_covers_both_categories():
    # The template is meant to be copied to eval/golden_set.json and edited, so it has to
    # parse as-is — including the `_comment_*` keys used to document it inline.
    examples = load_examples(TEMPLATE_PATH)
    categories = {example.category for example in examples}
    assert categories == {CATEGORY_SINGLE_PASSAGE, CATEGORY_CROSS_DOCUMENT}
    assert all(example.relevant_document_ids for example in examples)
    # Cross-document questions must name more than one document, or they are not testing the
    # thing the category exists for.
    cross = [e for e in examples if e.category == CATEGORY_CROSS_DOCUMENT]
    assert all(len(e.relevant_document_ids) >= 2 for e in cross)
