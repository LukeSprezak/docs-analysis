from app.knowledge_management.application.use_cases.upload_document import UploadDocumentUseCase
from app.knowledge_management.domain.models import Document


class FakeDocRepo:
    def __init__(self):
        self.saved = None

    async def save(self, document, owner_id):
        self.saved = (document, owner_id)


class FakeVectorRepo:
    def __init__(self):
        self.added = None

    async def add_documents(self, documents, owner_id):
        self.added = (documents, owner_id)


async def test_upload_namespaces_document_id_and_saves_whole_document_without_pages():
    doc_repo = FakeDocRepo()
    vector_repo = FakeVectorRepo()

    document = await UploadDocumentUseCase(doc_repo, vector_repo).execute(
        doc_id="a.txt", content="treść", metadata={"filename": "a.txt"}, owner_id="o1"
    )

    # Id namespace'owany per użytkownik (anty-kolizja nazw między userami).
    assert document.id == "o1::a.txt"
    saved_document, saved_owner = doc_repo.saved
    assert saved_document.id == "o1::a.txt"
    assert saved_owner == "o1"
    # Bez pages cały dokument trafia do wektorów (chunking dzieje się w repo wektorowym).
    added_documents, added_owner = vector_repo.added
    assert added_owner == "o1"
    assert len(added_documents) == 1
    assert added_documents[0].id == "o1::a.txt"


async def test_upload_with_pages_carries_page_numbers_and_namespaced_id():
    doc_repo = FakeDocRepo()
    vector_repo = FakeVectorRepo()
    pages = [
        Document(id="ignorowane", content="strona 1", metadata={"page": 1}),
        Document(id="ignorowane", content="strona 2", metadata={"page": 2}),
    ]

    await UploadDocumentUseCase(doc_repo, vector_repo).execute(
        doc_id="r.pdf",
        content="całość",
        metadata={"filename": "r.pdf"},
        owner_id="o1",
        pages=pages,
    )

    added_documents, _ = vector_repo.added
    # Każda strona idzie do wektorów z numerem strony i namespace'owanym id dokumentu.
    assert [document.metadata["page"] for document in added_documents] == [1, 2]
    assert all(document.id == "o1::r.pdf" for document in added_documents)
    assert all(document.metadata["filename"] == "r.pdf" for document in added_documents)
