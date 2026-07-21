from app.knowledge_management.application.use_cases.upload_document import UploadDocumentUseCase
from app.knowledge_management.domain.models import Document
from tests.fakes import StubDocumentRepo, StubVectorStoreRepo


class FakeDocRepo(StubDocumentRepo):
    def __init__(self) -> None:
        self.saved: tuple[Document, str] | None = None

    async def save(self, document: Document, owner_id: str) -> None:
        self.saved = (document, owner_id)


class FakeVectorRepo(StubVectorStoreRepo):
    def __init__(self) -> None:
        self.added: tuple[list[Document], str] | None = None

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        self.added = (documents, owner_id)


async def test_upload_namespaces_document_id_and_saves_whole_document_without_pages():
    doc_repo = FakeDocRepo()
    vector_repo = FakeVectorRepo()

    document = await UploadDocumentUseCase(doc_repo, vector_repo).execute(
        doc_id="a.txt", content="content", metadata={"filename": "a.txt"}, owner_id="o1"
    )

    # The id is namespaced per user (prevents name collisions between users).
    assert document.id == "o1::a.txt"
    assert doc_repo.saved is not None
    saved_document, saved_owner = doc_repo.saved
    assert saved_document.id == "o1::a.txt"
    assert saved_owner == "o1"
    # Without pages the whole document goes into the vectors (chunking happens in the vector repo).
    assert vector_repo.added is not None
    added_documents, added_owner = vector_repo.added
    assert added_owner == "o1"
    assert len(added_documents) == 1
    assert added_documents[0].id == "o1::a.txt"


async def test_upload_with_pages_carries_page_numbers_and_namespaced_id():
    doc_repo = FakeDocRepo()
    vector_repo = FakeVectorRepo()
    pages = [
        Document(id="ignored", content="page 1", metadata={"page": 1}),
        Document(id="ignored", content="page 2", metadata={"page": 2}),
    ]

    await UploadDocumentUseCase(doc_repo, vector_repo).execute(
        doc_id="r.pdf",
        content="whole document",
        metadata={"filename": "r.pdf"},
        owner_id="o1",
        pages=pages,
    )

    assert vector_repo.added is not None
    added_documents, _ = vector_repo.added
    # Every page goes into the vectors with its page number and the namespaced document id.
    assert [document.metadata["page"] for document in added_documents] == [1, 2]
    assert all(document.id == "o1::r.pdf" for document in added_documents)
    assert all(document.metadata["filename"] == "r.pdf" for document in added_documents)
