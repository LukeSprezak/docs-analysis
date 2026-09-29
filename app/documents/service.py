import os
from typing import Any

from anyio import to_thread

from app.documents.repo import DocumentRepo
from app.retrieval.api import IndexDocumentUseCase, RemoveFromIndexUseCase
from app.shared.kernel.document import Document
from app.shared.kernel.document_identity import namespaced_document_id
from app.shared.storage import is_within_storage


async def upload_document(
    doc_repo: DocumentRepo,
    index_document: IndexDocumentUseCase,
    doc_id: str,
    content: str,
    metadata: dict[str, Any],
    owner_id: str,
    pages: list[Document] | None = None,
) -> Document:
    # Namespaced per user so two users can upload a file with the same name without
    # colliding on the key; the rule itself lives in the domain. The original name stays
    # in the metadata ("filename") for display.
    document_id = namespaced_document_id(owner_id, doc_id)
    document = Document(id=document_id, content=content, metadata=metadata)
    await doc_repo.save(document, owner_id)
    await index_document.execute(document, owner_id, pages)
    return document


def _remove_file_if_within_storage(file_path: str) -> None:
    # Only delete files inside the storage directory — protects against removing an arbitrary
    # file via crafted metadata (legacy). Synchronous I/O — called in a thread so it does not
    # block the event loop (ASYNC240).
    if is_within_storage(file_path) and os.path.exists(file_path):
        os.remove(file_path)


async def delete_document(
    doc_repo: DocumentRepo, remove_from_index: RemoveFromIndexUseCase, doc_id: str, owner_id: str
) -> None:
    # get_by_id is filtered by owner_id — someone else's document is never found, so we
    # touch neither the file nor the vectors.
    doc = await doc_repo.get_by_id(doc_id, owner_id)
    if doc is None:
        return
    if "file_path" in doc.metadata:
        await to_thread.run_sync(_remove_file_if_within_storage, doc.metadata["file_path"])

    await remove_from_index.execute(doc_id, owner_id)
    await doc_repo.delete(doc_id, owner_id)
