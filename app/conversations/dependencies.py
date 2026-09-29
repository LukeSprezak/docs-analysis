from typing import Annotated

from fastapi import Depends

from app.conversations.chat import ChatWithDocsUseCase
from app.conversations.repo import ConversationRepo, PostgresConversationRepo
from app.retrieval.api import RAGService, RetrievalPipeline, get_rag_service, get_retrieval_pipeline


def get_conversation_repo() -> ConversationRepo:
    return PostgresConversationRepo()


def get_chat_with_docs_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
    retrieval_pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatWithDocsUseCase:
    return ChatWithDocsUseCase(conversation_repo, retrieval_pipeline, rag_service)
