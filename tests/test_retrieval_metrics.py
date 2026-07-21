from app.knowledge_management.application.evaluation import retrieval_metrics


def test_is_hit_at_k_true_when_relevant_in_top_k():
    assert retrieval_metrics.is_hit_at_k(["a", "b", "c"], {"c"}, k=3) is True


def test_is_hit_at_k_false_when_relevant_outside_top_k():
    # trafny "c" jest dopiero na 3. pozycji, a patrzymy na top 2
    assert retrieval_metrics.is_hit_at_k(["a", "b", "c"], {"c"}, k=2) is False


def test_precision_at_k_counts_relevant_among_returned():
    # 2 z 4 zwróconych są trafne
    assert retrieval_metrics.precision_at_k(["a", "b", "x", "y"], {"a", "b"}, k=4) == 0.5


def test_precision_at_k_divides_by_returned_not_by_k():
    # zwrócono tylko 2 pozycje przy k=4 — dzielimy przez 2, nie przez 4
    assert retrieval_metrics.precision_at_k(["a", "b"], {"a", "b"}, k=4) == 1.0


def test_precision_at_k_empty_retrieved_is_zero():
    assert retrieval_metrics.precision_at_k([], {"a"}, k=4) == 0.0


def test_recall_at_k_fraction_of_relevant_covered():
    # z 2 trafnych pokryto 1 w top_k
    assert retrieval_metrics.recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_no_relevant_defined_is_zero():
    assert retrieval_metrics.recall_at_k(["a", "b"], set(), k=3) == 0.0


def test_recall_at_k_deduplicates_repeated_chunks_from_same_document():
    # ten sam dokument na kilku pozycjach liczy się raz
    assert retrieval_metrics.recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


def test_reciprocal_rank_uses_position_of_first_relevant():
    assert retrieval_metrics.reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5


def test_reciprocal_rank_zero_when_no_relevant_retrieved():
    assert retrieval_metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_of_empty_is_zero():
    assert retrieval_metrics.mean([]) == 0.0


def test_mean_basic():
    assert retrieval_metrics.mean([1.0, 0.0, 0.5]) == 0.5