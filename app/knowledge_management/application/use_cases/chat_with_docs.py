from ...domain.repositories import VectorStoreRepo, RAGService
from ...domain.models import Answer

class ChatWithDocsUseCase:
    def __init__(self, vector_repo: VectorStoreRepo, rag_service: RAGService):
        self.vector_repo = vector_repo
        self.rag_service = rag_service

    def execute(self, message: str, history: list = None) -> Answer:
        relevant_docs = self.vector_repo.search(message)
        answer_text = self.rag_service.answer_question(message, relevant_docs)
        return Answer(text=answer_text, sources=relevant_docs)
