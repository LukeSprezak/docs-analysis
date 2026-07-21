from app.knowledge_management.application.use_cases.summarize_docs import SummarizeDocsUseCase
from app.knowledge_management.domain.models import Document, Summary
from app.knowledge_management.domain.repositories import SummarizerService
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
        self.saved: tuple[Summary, str] | None = None

    async def save(self, summary: Summary, owner_id: str) -> str:
        self.saved = (summary, owner_id)
        return summary.id or "s1"


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
    # The text comes from the summarizer, but document_ids preserves the original input.
    assert summary.text == "summary"
    assert summary.document_ids == ["a", "missing", "b"]
    assert summary_repo.saved is not None
    saved_summary, saved_owner = summary_repo.saved
    assert saved_owner == "o1"
    assert saved_summary.text == "summary"
