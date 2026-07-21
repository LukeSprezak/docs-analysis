from ...domain.models import Document
from ...domain.repositories import RerankerService, VectorStoreRepo


class RetrievalPipeline:
    """The shared `search(N candidates) → rerank → top_k` step.

    Mirrors exactly the pattern the QA/chat use cases follow, so the evaluation measures the
    retrieval that really reaches production. Shared by the retrieval and the generation
    evaluator (DRY).
    """

    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        reranker: RerankerService,
        candidate_count: int = 20,
        top_k: int = 4,
    ) -> None:
        self.vector_repo = vector_repo
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.top_k = top_k

    async def retrieve(self, query: str, owner_id: str) -> list[Document]:
        candidates = await self.vector_repo.search(query, owner_id, top_k=self.candidate_count)
        return await self.reranker.rerank(query, candidates, top_k=self.top_k)
