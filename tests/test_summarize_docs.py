from app.knowledge_management.application.use_cases.summarize_docs import SummarizeDocsUseCase
from app.knowledge_management.domain.models import Document


class FakeDocRepo:
    def __init__(self, documents_by_id):
        self._documents_by_id = documents_by_id
        self.get_calls = []

    async def get_by_id(self, doc_id, owner_id):
        self.get_calls.append((doc_id, owner_id))
        return self._documents_by_id.get(doc_id)


class FakeSummarizer:
    def __init__(self):
        self.received_documents = None

    async def summarize(self, documents):
        self.received_documents = documents
        return "summary"


class FakeSummaryRepo:
    def __init__(self):
        self.saved = None

    async def save(self, summary, owner_id):
        self.saved = (summary, owner_id)


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
    saved_summary, saved_owner = summary_repo.saved
    assert saved_owner == "o1"
    assert saved_summary.text == "summary"
