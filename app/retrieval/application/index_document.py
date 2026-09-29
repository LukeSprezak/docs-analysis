from app.retrieval.domain.null_entity_extractor import NullEntityExtractor
from app.retrieval.domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from app.retrieval.domain.repositories import EntityExtractor, KnowledgeGraphRepo, VectorStoreRepo
from app.shared.kernel.document import Document


class IndexDocumentUseCase:
    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        graph_repo: KnowledgeGraphRepo | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo or NullKnowledgeGraphRepo()
        self.entity_extractor = entity_extractor or NullEntityExtractor()

    async def execute(
        self, document: Document, owner_id: str, pages: list[Document] | None = None
    ) -> None:
        # What goes into the vector store is either the pages (if the loader split them),
        # keeping the page number, or the whole document. Chunking into smaller pieces
        # happens further down, in the vector repository.
        if pages:
            vector_docs = [
                Document(
                    id=document.id,
                    content=page.content,
                    metadata={**document.metadata, "page": page.metadata.get("page")},
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
