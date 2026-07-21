import os
import tempfile
from pathlib import Path

from app.knowledge_management.application.use_cases.delete_document import DeleteDocumentUseCase
from app.knowledge_management.domain.models import Document
from app.shared.storage import STORAGE_DOCUMENTS_DIR


class FakeDocRepo:
    def __init__(self, document):
        self._document = document
        self.deleted = []
        self.get_args = None

    async def get_by_id(self, doc_id, owner_id):
        self.get_args = (doc_id, owner_id)
        return self._document

    async def delete(self, doc_id, owner_id):
        self.deleted.append((doc_id, owner_id))


class FakeVectorRepo:
    def __init__(self):
        self.deleted = []

    async def delete_by_document_id(self, document_id, owner_id):
        self.deleted.append((document_id, owner_id))


async def test_delete_own_document_removes_file_and_vectors():
    os.makedirs(STORAGE_DOCUMENTS_DIR, exist_ok=True)
    file_descriptor, file_path = tempfile.mkstemp(dir=STORAGE_DOCUMENTS_DIR)
    os.close(file_descriptor)
    document = Document(id="o1::a.txt", content="x", metadata={"file_path": file_path})
    doc_repo = FakeDocRepo(document)
    vector_repo = FakeVectorRepo()

    await DeleteDocumentUseCase(doc_repo, vector_repo).execute("o1::a.txt", "o1")

    assert not Path(file_path).exists()  # File inside storage deleted
    assert vector_repo.deleted == [("o1::a.txt", "o1")]
    assert doc_repo.deleted == [("o1::a.txt", "o1")]


async def test_delete_missing_or_foreign_document_is_noop():
    # get_by_id filters by owner_id — someone else's document is never returned → None.
    doc_repo = FakeDocRepo(None)
    vector_repo = FakeVectorRepo()

    await DeleteDocumentUseCase(doc_repo, vector_repo).execute("o1::a.txt", "intruder")

    assert doc_repo.get_args == ("o1::a.txt", "intruder")
    assert vector_repo.deleted == []
    assert doc_repo.deleted == []


async def test_delete_does_not_remove_file_outside_storage():
    # The metadata points to a file OUTSIDE the storage — the guard must not touch it.
    file_descriptor, outside_path = tempfile.mkstemp()
    os.close(file_descriptor)
    try:
        document = Document(id="o1::a.txt", content="x", metadata={"file_path": outside_path})
        doc_repo = FakeDocRepo(document)
        vector_repo = FakeVectorRepo()

        await DeleteDocumentUseCase(doc_repo, vector_repo).execute("o1::a.txt", "o1")

        # NOT deleted (protection against removing other people's files)
        assert Path(outside_path).exists()
        assert vector_repo.deleted == [("o1::a.txt", "o1")]
        assert doc_repo.deleted == [("o1::a.txt", "o1")]
    finally:
        os.remove(outside_path)


async def test_delete_without_file_path_skips_file_removal():
    document = Document(id="o1::a.txt", content="x", metadata={})  # No path to the file
    doc_repo = FakeDocRepo(document)
    vector_repo = FakeVectorRepo()

    await DeleteDocumentUseCase(doc_repo, vector_repo).execute("o1::a.txt", "o1")

    assert vector_repo.deleted == [("o1::a.txt", "o1")]
    assert doc_repo.deleted == [("o1::a.txt", "o1")]
