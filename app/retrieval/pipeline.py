from app.retrieval.knowledge_graph.null import NullKnowledgeGraphRepo
from app.retrieval.ports import KnowledgeGraphRepo, RerankerService, VectorStoreRepo
from app.retrieval.rank_fusion import fuse_documents, retrieval_key
from app.shared.kernel.document import Document


class RetrievalPipeline:
    """The shared `search(N candidates) → rerank → top_k` step.

    Used by the QA and chat endpoints and by both evaluators, so the evaluation measures the
    retrieval that really reaches production.

    Candidates come from the vector store plus, when a knowledge graph is configured, the facts
    connected to whatever the question mentions; the two rankings are fused with RRF. There is
    no `if graph_enabled`: with the graph off the repository is the null one, its result is
    empty, and fusing a ranking with an empty one returns the first ranking in its original
    order. "Graph disabled" and "graph found nothing" are then the same code path.

    `graph_repo` is what makes a vector-only vs vector+graph comparison possible: build two
    pipelines over the same corpus, one with the null graph and one with the real one, and the
    metric difference is the graph's contribution. Everything else stays identical, so nothing
    but the extra candidate source can explain a change in the numbers.
    """

    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        reranker: RerankerService,
        candidate_count: int = 20,
        top_k: int = 4,
        graph_repo: KnowledgeGraphRepo | None = None,
        min_score: float = 0.0,
    ) -> None:
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo or NullKnowledgeGraphRepo()
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.top_k = top_k
        self.min_score = min_score

    async def retrieve(self, query: str, owner_id: str) -> list[Document]:
        passages = await self.vector_repo.search(query, owner_id, top_k=self.candidate_count)
        facts = await self.graph_repo.search_related(query, owner_id, top_k=self.candidate_count)
        candidates = fuse_documents(
            [passages, facts], top_k=self.candidate_count, key_of=retrieval_key
        )
        # Cut before reranking, so weak hits never take a top_k slot. Keyword-only hits and
        # graph facts have no similarity score to compare — they matched on their own terms.
        candidates = [
            candidate
            for candidate in candidates
            if "score" not in candidate.metadata or candidate.metadata["score"] >= self.min_score
        ]
        return await self.reranker.rerank(query, candidates, top_k=self.top_k)
