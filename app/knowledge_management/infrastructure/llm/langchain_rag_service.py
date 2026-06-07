from typing import List
from ...domain.repositories import RAGService
from ...domain.models import Document

class LangChainRAGService(RAGService):
    def answer_question(self, question: str, context: List[Document]) -> str:
        context_text = "\n".join([doc.content for doc in context])
        return f"Odpowiedź na pytanie '{question}' na podstawie kontekstu o długości {len(context_text)} znaków."
