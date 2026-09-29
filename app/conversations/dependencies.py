import threading
from typing import Annotated

from fastapi import Depends

from app.conversations.application.chat_with_docs import ChatWithDocsUseCase
from app.conversations.application.manage_conversations import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
)
from app.conversations.domain.repositories import ConversationRepo
from app.conversations.infrastructure import factory
from app.retrieval.api import RAGService, RetrievalPipeline, get_rag_service, get_retrieval_pipeline

_singleton_lock = threading.Lock()

_conversation_repo: ConversationRepo | None = None


def get_conversation_repo() -> ConversationRepo:
    global _conversation_repo
    with _singleton_lock:
        if _conversation_repo is None:
            _conversation_repo = factory.create_conversation_repo()
        return _conversation_repo


def init() -> None:
    get_conversation_repo()


async def shutdown() -> None:
    # Nothing to close: the repository borrows the shared pool that `dispose_engine()` closes.
    global _conversation_repo
    _conversation_repo = None


def get_chat_with_docs_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
    retrieval_pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatWithDocsUseCase:
    return ChatWithDocsUseCase(conversation_repo, retrieval_pipeline, rag_service)


def get_list_conversations_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
) -> ListConversationsUseCase:
    return ListConversationsUseCase(conversation_repo)


def get_get_conversation_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
) -> GetConversationUseCase:
    return GetConversationUseCase(conversation_repo)


def get_delete_conversation_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
) -> DeleteConversationUseCase:
    return DeleteConversationUseCase(conversation_repo)
