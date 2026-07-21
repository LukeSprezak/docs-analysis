from app.knowledge_management.application.evaluation.evaluate_generation import (
    GenerationEvaluator,
)
from app.knowledge_management.application.evaluation.retrieval_pipeline import RetrievalPipeline
from app.knowledge_management.domain.evaluation import EvaluationExample
from app.knowledge_management.domain.models import Document


class FakeVectorStoreRepo:
    def __init__(self, documents):
        self._documents = documents

    async def search(self, query, owner_id, top_k=4):
        return self._documents

    async def add_documents(self, documents, owner_id):  # pragma: no cover
        raise NotImplementedError

    async def delete_by_document_id(self, document_id, owner_id):  # pragma: no cover
        raise NotImplementedError


class PassthroughReranker:
    async def rerank(self, query, documents, top_k):
        return documents[:top_k]


class FakeRAGService:
    def __init__(self):
        self.received_context = None

    async def answer_question(self, question, context, history=None):
        self.received_context = context
        return f"odpowiedź na: {question}"

    async def condense_question(self, question, history):  # pragma: no cover
        return question

    async def astream_answer(self, question, context, history=None):  # pragma: no cover
        raise NotImplementedError


class FakeJudge:
    """It returns the final grades and stores the information it received for grading."""

    def __init__(self, faithfulness, answer_relevance):
        self._faithfulness = faithfulness
        self._answer_relevance = answer_relevance
        self.faithfulness_context = None
        self.relevance_args = None

    async def score_faithfulness(self, answer, context):
        self.faithfulness_context = context
        return self._faithfulness

    async def score_answer_relevance(self, question, answer):
        self.relevance_args = (question, answer)
        return self._answer_relevance


def _build(documents, rag_service, judge, top_k=4):
    pipeline = RetrievalPipeline(
        FakeVectorStoreRepo(documents), PassthroughReranker(), candidate_count=20, top_k=top_k
    )
    return GenerationEvaluator(pipeline, rag_service, judge)


async def test_evaluate_example_generates_answer_and_scores_it():
    documents = [Document(id="d", content="kontekst", metadata={})]
    rag_service = FakeRAGService()
    judge = FakeJudge(faithfulness=0.9, answer_relevance=0.8)
    evaluator = _build(documents, rag_service, judge)

    result = await evaluator.evaluate_example(
        EvaluationExample(question="Question?", relevant_document_ids=["d"]), owner_id="owner1"
    )

    assert result.answer == "Response to: Question?"
    assert result.faithfulness == 0.9
    assert result.answer_relevance == 0.8
    # The “faithfulness” metric receives a sophisticated context and relevance—a question and an answer
    assert judge.faithfulness_context == documents
    assert judge.relevance_args == ("Question?", "Response to: Question?")


async def test_evaluate_aggregates_generation_metrics():
    documents = [Document(id="d", content="kontekst", metadata={})]
    evaluator = _build(documents, FakeRAGService(), FakeJudge(0.6, 1.0))
    examples = [
        EvaluationExample(question="A", relevant_document_ids=["d"]),
        EvaluationExample(question="B", relevant_document_ids=["d"]),
    ]

    metrics, details = await evaluator.evaluate(examples, owner_id="owner1")

    assert metrics.example_count == 2
    assert metrics.mean_faithfulness == 0.6
    assert metrics.mean_answer_relevance == 1.0
    assert len(details) == 2