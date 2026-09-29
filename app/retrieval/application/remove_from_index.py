from app.retrieval.domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from app.retrieval.domain.repositories import KnowledgeGraphRepo, VectorStoreRepo


class RemoveFromIndexUseCase:
    def __init__(self, vector_repo: VectorStoreRepo, graph_repo: KnowledgeGraphRepo | None = None):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo or NullKnowledgeGraphRepo()

    async def execute(self, doc_id: str, owner_id: str) -> None:
        await self.vector_repo.delete_by_document_id(doc_id, owner_id)
        # Retract this document's facts too, or the graph keeps answering from a document the
        # user believes they deleted.
        await self.graph_repo.delete_by_document_id(doc_id, owner_id)
