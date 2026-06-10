from ...domain.models import Answer
from ...domain.repositories import RAGService, RerankerService, VectorStoreRepo


class AskQuestionUseCase:
    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        rag_service: RAGService,
        reranker: RerankerService,
        candidate_count: int = 20,
        top_k: int = 4,
    ):
        self.vector_repo = vector_repo
        self.rag_service = rag_service
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.top_k = top_k

    async def execute(self, question_text: str, owner_id: str) -> Answer:
        candidates = await self.vector_repo.search(
            question_text, owner_id, top_k=self.candidate_count
        )
        relevant_docs = await self.reranker.rerank(
            question_text, candidates, top_k=self.top_k
        )
        answer_text = await self.rag_service.answer_question(question_text, relevant_docs)

        return Answer(text=answer_text, sources=relevant_docs)
