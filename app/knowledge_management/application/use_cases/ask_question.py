from ...domain.repositories import VectorStoreRepo, RAGService
from ...domain.models import Query, Answer

class AskQuestionUseCase:
    def __init__(self, vector_repo: VectorStoreRepo, rag_service: RAGService):
        self.vector_repo = vector_repo
        self.rag_service = rag_service

    def execute(self, query_text: str) -> Answer:
        relevant_docs = self.vector_repo.search(query_text)
        answer_text = self.rag_service.answer_question(query_text, relevant_docs)
        return Answer(text=answer_text, sources=relevant_docs)
