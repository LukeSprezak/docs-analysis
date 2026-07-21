import os

from anyio import to_thread

from app.shared.storage import is_within_storage

from ...domain.repositories import DocumentRepo, VectorStoreRepo


def _remove_file_if_within_storage(file_path: str) -> None:
    # Only delete files inside the storage directory — protects against removing an arbitrary
    # file via crafted metadata (legacy). Synchronous I/O — called in a thread so it does not
    # block the event loop (ASYNC240).
    if is_within_storage(file_path) and os.path.exists(file_path):
        os.remove(file_path)


class DeleteDocumentUseCase:
    def __init__(self, doc_repo: DocumentRepo, vector_repo: VectorStoreRepo):
        self.doc_repo = doc_repo
        self.vector_repo = vector_repo

    async def execute(self, doc_id: str, owner_id: str) -> None:
        # get_by_id is filtered by owner_id — someone else's document is never found, so we
        # touch neither the file nor the vectors.
        doc = await self.doc_repo.get_by_id(doc_id, owner_id)
        if doc is None:
            return
        if "file_path" in doc.metadata:
            await to_thread.run_sync(_remove_file_if_within_storage, doc.metadata["file_path"])

        await self.vector_repo.delete_by_document_id(doc_id, owner_id)
        await self.doc_repo.delete(doc_id, owner_id)
