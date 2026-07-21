from app.knowledge_management.application.evaluation import retrieval_metrics


def test_is_hit_at_k_true_when_relevant_in_top_k():
    assert retrieval_metrics.is_hit_at_k(["a", "b", "c"], {"c"}, k=3) is True


def test_is_hit_at_k_false_when_relevant_outside_top_k():
    # the relevant "c" sits only at position 3, and we look at the top 2
    assert retrieval_metrics.is_hit_at_k(["a", "b", "c"], {"c"}, k=2) is False


def test_precision_at_k_counts_relevant_among_returned():
    # 2 of the 4 returned items are relevant
    assert retrieval_metrics.precision_at_k(["a", "b", "x", "y"], {"a", "b"}, k=4) == 0.5


def test_precision_at_k_divides_by_returned_not_by_k():
    # only 2 items were returned with k=4 — we divide by 2, not by 4
    assert retrieval_metrics.precision_at_k(["a", "b"], {"a", "b"}, k=4) == 1.0


def test_precision_at_k_empty_retrieved_is_zero():
    assert retrieval_metrics.precision_at_k([], {"a"}, k=4) == 0.0


def test_recall_at_k_fraction_of_relevant_covered():
    # 1 of the 2 relevant items is covered within top_k
    assert retrieval_metrics.recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_no_relevant_defined_is_zero():
    assert retrieval_metrics.recall_at_k(["a", "b"], set(), k=3) == 0.0


def test_recall_at_k_deduplicates_repeated_chunks_from_same_document():
    # the same document at several positions counts once
    assert retrieval_metrics.recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


def test_reciprocal_rank_uses_position_of_first_relevant():
    assert retrieval_metrics.reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5


def test_reciprocal_rank_zero_when_no_relevant_retrieved():
    assert retrieval_metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_of_empty_is_zero():
    assert retrieval_metrics.mean([]) == 0.0


def test_mean_basic():
    assert retrieval_metrics.mean([1.0, 0.0, 0.5]) == 0.5
