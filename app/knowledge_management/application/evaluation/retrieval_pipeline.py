from ...domain.models import Document
from ...domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from ...domain.repositories import KnowledgeGraphRepo, RerankerService, VectorStoreRepo
from ..retrieval.candidate_retrieval import CandidateRetriever


class RetrievalPipeline:
    """The shared `search(N candidates) → rerank → top_k` step.

    Mirrors exactly the pattern the QA/chat use cases follow, so the evaluation measures the
    retrieval that really reaches production. Shared by the retrieval and the generation
    evaluator (DRY).

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
    ) -> None:
        self.retriever = CandidateRetriever(vector_repo, graph_repo or NullKnowledgeGraphRepo())
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.top_k = top_k

    async def retrieve(self, query: str, owner_id: str) -> list[Document]:
        candidates = await self.retriever.retrieve(
            query, owner_id, candidate_count=self.candidate_count
        )
        return await self.reranker.rerank(query, candidates, top_k=self.top_k)
