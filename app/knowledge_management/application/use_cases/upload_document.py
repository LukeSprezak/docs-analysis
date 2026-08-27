from typing import Any

from ...domain.document_identity import namespaced_document_id
from ...domain.models import Document
from ...domain.null_entity_extractor import NullEntityExtractor
from ...domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from ...domain.repositories import (
    DocumentRepo,
    EntityExtractor,
    KnowledgeGraphRepo,
    VectorStoreRepo,
)


class UploadDocumentUseCase:
    def __init__(
        self,
        doc_repo: DocumentRepo,
        vector_repo: VectorStoreRepo,
        graph_repo: KnowledgeGraphRepo | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        self.doc_repo = doc_repo
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo or NullKnowledgeGraphRepo()
        self.entity_extractor = entity_extractor or NullEntityExtractor()

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

        # What goes into the vector store is either the pages (if the loader split them),
        # keeping the page number, or the whole document. Chunking into smaller pieces
        # happens further down, in the vector repository.
        if pages:
            vector_docs = [
                Document(
                    id=document_id,
                    content=page.content,
                    metadata={**metadata, "page": page.metadata.get("page")},
                )
                for page in pages
            ]
        else:
            vector_docs = [document]
        await self.vector_repo.add_documents(vector_docs, owner_id)

        # Feed the knowledge graph from the whole document, not the per-page split: relations
        # regularly span a page boundary, and the extractor needs the surrounding text to see
        # them. With no graph configured both calls are no-ops and cost nothing.
        fragment = await self.entity_extractor.extract(document)
        await self.graph_repo.add_fragment(fragment, owner_id)
        return document
