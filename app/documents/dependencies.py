import threading
from typing import Annotated

from fastapi import Depends

from app.documents.application.delete_document import DeleteDocumentUseCase
from app.documents.application.upload_document import UploadDocumentUseCase
from app.documents.domain.repositories import DocumentRepo
from app.documents.infrastructure import factory
from app.retrieval.api import (
    IndexDocumentUseCase,
    RemoveFromIndexUseCase,
    get_index_document_use_case,
    get_remove_from_index_use_case,
)

_singleton_lock = threading.Lock()

_doc_repo: DocumentRepo | None = None


def get_doc_repo() -> DocumentRepo:
    global _doc_repo
    with _singleton_lock:
        if _doc_repo is None:
            _doc_repo = factory.create_document_repo()
        return _doc_repo


def init() -> None:
    get_doc_repo()


async def shutdown() -> None:
    # Nothing to close: the repository borrows the shared pool that `dispose_engine()` closes.
    global _doc_repo
    _doc_repo = None


def get_upload_document_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    index_document: Annotated[IndexDocumentUseCase, Depends(get_index_document_use_case)],
) -> UploadDocumentUseCase:
    return UploadDocumentUseCase(doc_repo, index_document)


def get_delete_document_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    remove_from_index: Annotated[RemoveFromIndexUseCase, Depends(get_remove_from_index_use_case)],
) -> DeleteDocumentUseCase:
    return DeleteDocumentUseCase(doc_repo, remove_from_index)
