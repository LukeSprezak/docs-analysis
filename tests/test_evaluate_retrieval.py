from app.knowledge_management.application.evaluation.evaluate_retrieval import RetrievalEvaluator
from app.knowledge_management.application.evaluation.retrieval_pipeline import RetrievalPipeline
from app.knowledge_management.domain.evaluation import EvaluationExample
from app.knowledge_management.domain.models import Document


class FakeVectorStoreRepo:
    """Returns the prepared fragments for each query; stores the call parameters."""

    def __init__(self, documents_by_query):
        self._documents_by_query = documents_by_query
        self.search_top_k = None
        self.search_owner_id = None

    async def search(self, query, owner_id, top_k=4):
        self.search_top_k = top_k
        self.search_owner_id = owner_id
        return self._documents_by_query.get(query, [])

    async def add_documents(self, documents, owner_id):  # pragma: no cover
        raise NotImplementedError

    async def delete_by_document_id(self, document_id, owner_id):  # pragma: no cover
        raise NotImplementedError


class PassthroughReranker:
    async def rerank(self, query, documents, top_k):
        return documents[:top_k]


def _chunk(filename: str) -> Document:
    return Document(id=f"owner1::{filename}", content="...", metadata={"filename": filename})


def _build_evaluator(documents_by_query, candidate_count=20, top_k=4):
    vector_repo = FakeVectorStoreRepo(documents_by_query)
    pipeline = RetrievalPipeline(
        vector_repo, PassthroughReranker(), candidate_count=candidate_count, top_k=top_k
    )
    return RetrievalEvaluator(pipeline), vector_repo


async def test_evaluate_example_extracts_filename_and_scores_hit():
    evaluator, _ = _build_evaluator({"Q": [_chunk("good.pdf"), _chunk("other.pdf")]}, top_k=2)
    example = EvaluationExample(question="Q", relevant_document_ids=["good.pdf"])

    result = await evaluator.evaluate_example(example, owner_id="owner1")

    assert result.retrieved_document_ids == ["good.pdf", "other.pdf"]
    assert result.is_hit is True
    assert result.reciprocal_rank == 1.0
    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 1.0


async def test_evaluate_propagates_candidate_count_and_owner_to_search():
    evaluator, vector_repo = _build_evaluator(
        {"Q": [_chunk("good.pdf")]}, candidate_count=15, top_k=4
    )
    example = EvaluationExample(question="Q", relevant_document_ids=["good.pdf"])

    await evaluator.evaluate([example], owner_id="owner1")

    # Vector search receives a large set of candidates (candidate_count), not top_k
    assert vector_repo.search_top_k == 15
    assert vector_repo.search_owner_id == "owner1"


async def test_evaluate_aggregates_metrics_across_examples():
    documents_by_query = {
        "hit": [_chunk("good.pdf")],
        "miss": [_chunk("wrong.pdf")],
    }
    evaluator, _ = _build_evaluator(documents_by_query, top_k=4)
    examples = [
        EvaluationExample(question="hit", relevant_document_ids=["good.pdf"]),
        EvaluationExample(question="miss", relevant_document_ids=["good.pdf"]),
    ]

    metrics, details = await evaluator.evaluate(examples, owner_id="owner1")

    assert metrics.example_count == 2
    assert metrics.hit_rate == 0.5
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.mean_recall_at_k == 0.5
    assert len(details) == 2


async def test_evaluate_example_misses_when_no_relevant_retrieved():
    evaluator, _ = _build_evaluator({"Q": [_chunk("wrong.pdf")]}, top_k=4)
    example = EvaluationExample(question="Q", relevant_document_ids=["good.pdf"])

    result = await evaluator.evaluate_example(example, owner_id="owner1")

    assert result.is_hit is False
    assert result.reciprocal_rank == 0.0
    assert result.recall_at_k == 0.0
