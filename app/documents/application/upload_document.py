from typing import Any

from app.documents.domain.repositories import DocumentRepo
from app.retrieval.api import IndexDocumentUseCase
from app.shared.kernel.document import Document
from app.shared.kernel.document_identity import namespaced_document_id


class UploadDocumentUseCase:
    def __init__(self, doc_repo: DocumentRepo, index_document: IndexDocumentUseCase):
        self.doc_repo = doc_repo
        self.index_document = index_document

    async def execute(
        self,
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
        await self.doc_repo.save(document, owner_id)
        await self.index_document.execute(document, owner_id, pages)
        return document
