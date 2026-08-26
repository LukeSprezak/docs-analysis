"""Dependency injection for the knowledge_management context.

Every provider here is typed against the **port** (the ABC from `domain/repositories.py`),
never a concrete adapter. That is what keeps the backing store swappable: the use cases and
routers below are compiled against the contract, and the only code that names
`PostgresDocumentRepo` and friends is `infrastructure/persistence/factory.py`.

The repositories are process-wide singletons — they hold no per-request state, only a handle
to a shared connection pool (or, for FAISS, the in-memory index). They are built eagerly by
`init_repositories()` in the application lifespan, and lazily (under a lock) by the providers
below for callers that run without one, such as the evaluation harness.
"""

import threading
from typing import Annotated

from fastapi import Depends

from app.knowledge_management.application.use_cases.ask_question import AskQuestionUseCase
from app.knowledge_management.application.use_cases.chat_with_docs import ChatWithDocsUseCase
from app.knowledge_management.application.use_cases.delete_document import DeleteDocumentUseCase
from app.knowledge_management.application.use_cases.delete_summary import DeleteSummaryUseCase
from app.knowledge_management.application.use_cases.manage_conversations import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
)
from app.knowledge_management.application.use_cases.summarize_docs import SummarizeDocsUseCase
from app.knowledge_management.application.use_cases.upload_document import UploadDocumentUseCase
from app.knowledge_management.domain.repositories import (
    ConversationRepo,
    DocumentRepo,
    EntityExtractor,
    KnowledgeGraphRepo,
    RerankerService,
    SummaryRepo,
    VectorStoreRepo,
)
from app.knowledge_management.infrastructure.llm.langchain_rag_service import LangChainRAGService
from app.knowledge_management.infrastructure.llm.langchain_summarizer import LangChainSummarizer
from app.knowledge_management.infrastructure.llm.llm_factory import LLMFactory
from app.knowledge_management.infrastructure.llm.reranker_factory import RerankerFactory
from app.knowledge_management.infrastructure.persistence import factory
from app.shared.config import settings

_singleton_lock = threading.Lock()

_doc_repo: DocumentRepo | None = None
_vector_repo: VectorStoreRepo | None = None
_summary_repo: SummaryRepo | None = None
_conversation_repo: ConversationRepo | None = None
_graph_repo: KnowledgeGraphRepo | None = None
_entity_extractor: EntityExtractor | None = None


def get_doc_repo() -> DocumentRepo:
    global _doc_repo
    with _singleton_lock:
        if _doc_repo is None:
            _doc_repo = factory.create_document_repo()
        return _doc_repo


def get_vector_repo() -> VectorStoreRepo:
    global _vector_repo
    with _singleton_lock:
        if _vector_repo is None:
            _vector_repo = factory.create_vector_store_repo()
        return _vector_repo


def get_summary_repo() -> SummaryRepo:
    global _summary_repo
    with _singleton_lock:
        if _summary_repo is None:
            _summary_repo = factory.create_summary_repo()
        return _summary_repo


def get_conversation_repo() -> ConversationRepo:
    global _conversation_repo
    with _singleton_lock:
        if _conversation_repo is None:
            _conversation_repo = factory.create_conversation_repo()
        return _conversation_repo


def get_graph_repo() -> KnowledgeGraphRepo:
    global _graph_repo
    with _singleton_lock:
        if _graph_repo is None:
            _graph_repo = factory.create_knowledge_graph_repo()
        return _graph_repo


def get_entity_extractor() -> EntityExtractor:
    global _entity_extractor
    with _singleton_lock:
        if _entity_extractor is None:
            _entity_extractor = factory.create_entity_extractor()
        return _entity_extractor


def init_repositories() -> None:
    """Builds every repository singleton up front (called from the application lifespan).

    Pairs with `shutdown_repositories`: what the lifespan closes, the lifespan also opens —
    instead of the singletons appearing on whichever request happened to arrive first. A
    missing or unreachable Neo4j therefore fails the startup rather than the first query,
    which is the same reasoning `docker-compose.yml` uses to wait for the graph to be healthy.
    """
    get_doc_repo()
    get_vector_repo()
    get_summary_repo()
    get_conversation_repo()
    get_graph_repo()
    get_entity_extractor()


async def shutdown_repositories() -> None:
    """Releases the repository singletons (called from the application lifespan).

    The vector store and the knowledge graph can each own a driver; the record repositories
    all sit on the shared SQLAlchemy pool, which `dispose_engine()` closes separately. The
    singletons are cleared as well so a subsequent startup in the same process (tests, an ASGI
    reload) builds them fresh instead of handing out repos backed by a closed driver.
    """
    global _doc_repo, _vector_repo, _summary_repo, _conversation_repo
    global _graph_repo, _entity_extractor
    if _vector_repo is not None:
        await _vector_repo.close()
    if _graph_repo is not None:
        await _graph_repo.close()
    _doc_repo = _vector_repo = _summary_repo = _conversation_repo = None
    _graph_repo = _entity_extractor = None


def get_summarizer() -> LangChainSummarizer:
    llm = LLMFactory.get_llm()
    return LangChainSummarizer(llm=llm)


def get_rag_service() -> LangChainRAGService:
    llm = LLMFactory.get_llm()
    return LangChainRAGService(llm=llm)


def get_reranker_service() -> RerankerService:
    return RerankerFactory.get_reranker()


def get_upload_document_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
    entity_extractor: Annotated[EntityExtractor, Depends(get_entity_extractor)],
) -> UploadDocumentUseCase:
    return UploadDocumentUseCase(doc_repo, vector_repo, graph_repo, entity_extractor)


def get_summarize_docs_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    summarizer: Annotated[LangChainSummarizer, Depends(get_summarizer)],
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
) -> SummarizeDocsUseCase:
    return SummarizeDocsUseCase(doc_repo, summarizer, summary_repo)


def get_ask_question_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    rag_service: Annotated[LangChainRAGService, Depends(get_rag_service)],
    reranker: Annotated[RerankerService, Depends(get_reranker_service)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> AskQuestionUseCase:
    return AskQuestionUseCase(
        vector_repo,
        rag_service,
        reranker,
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
        graph_repo=graph_repo,
    )


def get_chat_with_docs_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    rag_service: Annotated[LangChainRAGService, Depends(get_rag_service)],
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
    reranker: Annotated[RerankerService, Depends(get_reranker_service)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> ChatWithDocsUseCase:
    return ChatWithDocsUseCase(
        vector_repo,
        rag_service,
        conversation_repo,
        reranker,
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
        graph_repo=graph_repo,
    )


def get_delete_document_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> DeleteDocumentUseCase:
    return DeleteDocumentUseCase(doc_repo, vector_repo, graph_repo)


def get_delete_summary_use_case(
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
) -> DeleteSummaryUseCase:
    return DeleteSummaryUseCase(summary_repo)


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
