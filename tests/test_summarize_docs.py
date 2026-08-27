import pytest

from app.knowledge_management.application.use_cases.summarize_docs import SummarizeDocsUseCase
from app.knowledge_management.domain.models import Document, Summary
from app.knowledge_management.domain.repositories import SummarizerService
from app.shared.exceptions import ValidationException
from tests.fakes import StubDocumentRepo, StubSummaryRepo


class FakeDocRepo(StubDocumentRepo):
    def __init__(self, documents_by_id: dict[str, Document]) -> None:
        self._documents_by_id = documents_by_id
        self.get_calls: list[tuple[str, str]] = []

    async def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        self.get_calls.append((doc_id, owner_id))
        return self._documents_by_id.get(doc_id)


class FakeSummarizer(SummarizerService):
    def __init__(self) -> None:
        self.received_documents: list[Document] | None = None

    async def summarize(self, documents: list[Document]) -> str:
        self.received_documents = documents
        return "summary"


class FakeSummaryRepo(StubSummaryRepo):
    def __init__(self) -> None:
        self.saved: tuple[str, list[str], str] | None = None

    async def save(self, text: str, document_ids: list[str], owner_id: str) -> Summary:
        self.saved = (text, document_ids, owner_id)
        return Summary(
            text=text, document_ids=document_ids, id="s1", created_at="2026-01-01T00:00:00"
        )


async def test_summarize_gathers_only_existing_owned_documents_and_saves():
    documents = {
        "a": Document(id="o1::a", content="A", metadata={}),
        "b": Document(id="o1::b", content="B", metadata={}),
    }
    doc_repo = FakeDocRepo(documents)
    summarizer = FakeSummarizer()
    summary_repo = FakeSummaryRepo()

    summary = await SummarizeDocsUseCase(doc_repo, summarizer, summary_repo).execute(
        ["a", "missing", "b"], owner_id="o1"
    )

    # get_by_id filters by owner; "missing" returns None and is skipped.
    assert summarizer.received_documents == [documents["a"], documents["b"]]
    assert [call[1] for call in doc_repo.get_calls] == ["o1", "o1", "o1"]
    # The text comes from the summarizer; document_ids records what was actually summarized,
    # not what was asked for — "missing" was skipped, so it is not part of the summary.
    assert summary.text == "summary"
    assert summary.document_ids == ["o1::a", "o1::b"]
    # The stored summary comes back with the identity the store minted.
    assert summary.id == "s1"
    assert summary_repo.saved == ("summary", ["o1::a", "o1::b"], "o1")


async def test_summarize_refuses_when_no_requested_document_was_found():
    """ERR-01: nothing to summarize is a client error, not an empty prompt.

    Summarizing an empty list is a paid LLM call with no content, and the summary it returns
    would be stored claiming documents it never saw."""
    doc_repo = FakeDocRepo({})
    summarizer = FakeSummarizer()
    summary_repo = FakeSummaryRepo()

    with pytest.raises(ValidationException):
        await SummarizeDocsUseCase(doc_repo, summarizer, summary_repo).execute(
            ["missing"], owner_id="o1"
        )

    assert summarizer.received_documents is None
    assert summary_repo.saved is None
