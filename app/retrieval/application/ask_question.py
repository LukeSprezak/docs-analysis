from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.domain.models import Answer
from app.retrieval.domain.repositories import RAGService


class AskQuestionUseCase:
    def __init__(self, retrieval_pipeline: RetrievalPipeline, rag_service: RAGService):
        self.retrieval_pipeline = retrieval_pipeline
        self.rag_service = rag_service

    async def execute(self, question_text: str, owner_id: str) -> Answer:
        # Retrieval is limited to the asker's own documents (owner_id).
        relevant_docs = await self.retrieval_pipeline.retrieve(question_text, owner_id)
        answer_text = await self.rag_service.answer_question(question_text, relevant_docs)

        return Answer(text=answer_text, sources=relevant_docs)
