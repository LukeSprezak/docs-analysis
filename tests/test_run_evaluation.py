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


class FakeVectorStoreRepo:
    async def search(self, query, owner_id, top_k=4):
        return [Document(id="d", content="context", metadata={"filename": "good.pdf"})]

    async def add_documents(self, documents, owner_id):  # pragma: no cover
        raise NotImplementedError

    async def delete_by_document_id(self, document_id, owner_id):  # pragma: no cover
        raise NotImplementedError


class PassthroughReranker:
    async def rerank(self, query, documents, top_k):
        return documents[:top_k]


class FakeRAGService:
    async def answer_question(self, question, context, history=None):
        return "answer"

    async def condense_question(self, question, history):  # pragma: no cover
        return question

    async def astream_answer(self, question, context, history=None):  # pragma: no cover
        raise NotImplementedError


class FakeJudge:
    async def score_faithfulness(self, answer, context):
        return 1.0

    async def score_answer_relevance(self, question, answer):
        return 0.5


def _pipeline():
    return RetrievalPipeline(FakeVectorStoreRepo(), PassthroughReranker(), 20, 4)


def _examples():
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