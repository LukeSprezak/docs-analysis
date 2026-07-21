from typing import Any

from ...domain.models import Document
from ...domain.repositories import DocumentRepo, VectorStoreRepo


class UploadDocumentUseCase:
    def __init__(self, doc_repo: DocumentRepo, vector_repo: VectorStoreRepo):
        self.doc_repo = doc_repo
        self.vector_repo = vector_repo

    async def execute(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        owner_id: str,
        pages: list[Document] | None = None,
    ) -> Document:
        # Namespaced per user: two users can upload a file with the same name without
        # colliding on the key (documents.id and the chunk id in the vectors). The original
        # name stays in the metadata ("filename") for display.
        namespaced_document_id = f"{owner_id}::{doc_id}"
        document = Document(id=namespaced_document_id, content=content, metadata=metadata)
        await self.doc_repo.save(document, owner_id)

        # What goes into the vector store is either the pages (if the loader split them),
        # keeping the page number, or the whole document. Chunking into smaller pieces
        # happens further down, in the vector repository.
        if pages:
            vector_docs = [
                Document(
                    id=namespaced_document_id,
                    content=page.content,
                    metadata={**metadata, "page": page.metadata.get("page")},
                )
                for page in pages
            ]
        else:
            vector_docs = [document]
        await self.vector_repo.add_documents(vector_docs, owner_id)
        return document
