from app.knowledge_management.application.evaluation.evaluate_generation import (
    GenerationEvaluator,
)
from app.knowledge_management.application.evaluation.evaluate_retrieval import RetrievalEvaluator
from app.knowledge_management.application.evaluation.retrieval_pipeline import RetrievalPipeline
from app.knowledge_management.application.evaluation.run_evaluation import (
    build_report,
    format_report,
)
from app.knowledge_management.domain.evaluation import EvaluationExample
from app.knowledge_management.domain.models import Document
from tests.fakes import PassthroughReranker, StubAnswerJudge, StubRAGService, StubVectorStoreRepo


class FakeVectorStoreRepo(StubVectorStoreRepo):
    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        return [Document(id="d", content="context", metadata={"filename": "good.pdf"})]


class FakeRAGService(StubRAGService):
    async def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return "answer"


class FakeJudge(StubAnswerJudge):
    async def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        return 1.0

    async def score_answer_relevance(self, question: str, answer: str) -> float:
        return 0.5


def _pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(FakeVectorStoreRepo(), PassthroughReranker(), 20, 4)


def _examples() -> list[EvaluationExample]:
    return [EvaluationExample(question="Q", relevant_document_ids=["good.pdf"])]


async def test_build_report_retrieval_only_when_no_judge():
    report = await build_report(_examples(), "owner1", RetrievalEvaluator(_pipeline()))

    assert report.retrieval.example_count == 1
    assert report.retrieval.hit_rate == 1.0
    assert report.generation is None
    assert report.generation_details == []


async def test_build_report_includes_generation_when_evaluator_given():
    generation_evaluator = GenerationEvaluator(_pipeline(), FakeRAGService(), FakeJudge())
    report = await build_report(
        _examples(), "owner1", RetrievalEvaluator(_pipeline()), generation_evaluator
    )

    assert report.generation is not None
    assert report.generation.mean_faithfulness == 1.0
    assert report.generation.mean_answer_relevance == 0.5
    assert len(report.generation_details) == 1


async def test_format_report_omits_generation_section_when_absent():
    report = await build_report(_examples(), "owner1", RetrievalEvaluator(_pipeline()))
    text = format_report(report)

    assert "RETRIEVAL:" in text
    assert "hit_rate@k" in text
    assert "GENERATION" not in text


async def test_format_report_includes_generation_section_when_present():
    generation_evaluator = GenerationEvaluator(_pipeline(), FakeRAGService(), FakeJudge())
    report = await build_report(
        _examples(), "owner1", RetrievalEvaluator(_pipeline()), generation_evaluator
    )
    text = format_report(report)

    assert "GENERATION (LLM-as-judge):" in text
    assert "faithfulness" in text
