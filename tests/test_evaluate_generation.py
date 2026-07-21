from app.knowledge_management.application.evaluation.evaluate_generation import (
    GenerationEvaluator,
)
from app.knowledge_management.application.evaluation.retrieval_pipeline import RetrievalPipeline
from app.knowledge_management.domain.evaluation import EvaluationExample
from app.knowledge_management.domain.models import Document
from tests.fakes import PassthroughReranker, StubAnswerJudge, StubRAGService, StubVectorStoreRepo


class FakeVectorStoreRepo(StubVectorStoreRepo):
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        return self._documents


class FakeRAGService(StubRAGService):
    def __init__(self) -> None:
        self.received_context: list[Document] | None = None

    async def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self.received_context = context
        return f"Answer to: {question}"


class FakeJudge(StubAnswerJudge):
    """Returns fixed scores and records what it was given to score."""

    def __init__(self, faithfulness: float, answer_relevance: float) -> None:
        self._faithfulness = faithfulness
        self._answer_relevance = answer_relevance
        self.faithfulness_context: list[Document] | None = None
        self.relevance_args: tuple[str, str] | None = None

    async def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        self.faithfulness_context = context
        return self._faithfulness

    async def score_answer_relevance(self, question: str, answer: str) -> float:
        self.relevance_args = (question, answer)
        return self._answer_relevance


def _build(
    documents: list[Document],
    rag_service: FakeRAGService,
    judge: FakeJudge,
    top_k: int = 4,
) -> GenerationEvaluator:
    pipeline = RetrievalPipeline(
        FakeVectorStoreRepo(documents), PassthroughReranker(), candidate_count=20, top_k=top_k
    )
    return GenerationEvaluator(pipeline, rag_service, judge)


async def test_evaluate_example_generates_answer_and_scores_it():
    documents = [Document(id="d", content="context", metadata={})]
    rag_service = FakeRAGService()
    judge = FakeJudge(faithfulness=0.9, answer_relevance=0.8)
    evaluator = _build(documents, rag_service, judge)

    result = await evaluator.evaluate_example(
        EvaluationExample(question="Question?", relevant_document_ids=["d"]), owner_id="owner1"
    )

    assert result.answer == "Answer to: Question?"
    assert result.faithfulness == 0.9
    assert result.answer_relevance == 0.8
    # faithfulness is scored against the retrieved context, relevance against question + answer
    assert judge.faithfulness_context == documents
    assert judge.relevance_args == ("Question?", "Answer to: Question?")


async def test_evaluate_aggregates_generation_metrics():
    documents = [Document(id="d", content="context", metadata={})]
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
