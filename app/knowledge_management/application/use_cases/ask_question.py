from ...domain.models import Answer
from ...domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from ...domain.repositories import (
    KnowledgeGraphRepo,
    RAGService,
    RerankerService,
    VectorStoreRepo,
)
from ..retrieval.candidate_retrieval import CandidateRetriever


class AskQuestionUseCase:
    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        rag_service: RAGService,
        reranker: RerankerService,
        candidate_count: int = 20,
        top_k: int = 4,
        graph_repo: KnowledgeGraphRepo | None = None,
    ):
        # Defaults to the null graph so a caller that does not use the feature (most tests)
        # need not know it exists. Production always passes one explicitly — see the factory.
        self.retriever = CandidateRetriever(vector_repo, graph_repo or NullKnowledgeGraphRepo())
        self.rag_service = rag_service
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.top_k = top_k

    async def execute(self, question_text: str, owner_id: str) -> Answer:
        # Fetch a wider candidate set, then reorder it down to the best top_k.
        # Retrieval is limited to the asker's own documents (owner_id).
        candidates = await self.retriever.retrieve(
            question_text, owner_id, candidate_count=self.candidate_count
        )
        relevant_docs = await self.reranker.rerank(question_text, candidates, top_k=self.top_k)
        answer_text = await self.rag_service.answer_question(question_text, relevant_docs)

        return Answer(text=answer_text, sources=relevant_docs)
