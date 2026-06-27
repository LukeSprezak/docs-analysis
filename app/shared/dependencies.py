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
from app.knowledge_management.domain.repositories import RerankerService, VectorStoreRepo
from app.knowledge_management.infrastructure.llm.embeddings_factory import EmbeddingsFactory
from app.knowledge_management.infrastructure.llm.langchain_rag_service import LangChainRAGService
from app.knowledge_management.infrastructure.llm.langchain_summarizer import LangChainSummarizer
from app.knowledge_management.infrastructure.llm.llm_factory import LLMFactory
from app.knowledge_management.infrastructure.llm.reranker_factory import RerankerFactory
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import (
    FaissVectorStoreRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_conversation_repo import (
    PostgresConversationRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_document_repo import (
    PostgresDocumentRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_summary_repo import (
    PostgresSummaryRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_vectorstore_repo import (
    PostgresVectorStoreRepo,
)
from app.shared.config import settings
from app.shared.enums import SearchStrategy, VectorStoreProvider

_doc_repo: PostgresDocumentRepo | None = None
_vector_repo: VectorStoreRepo | None = None
_summary_repo: PostgresSummaryRepo | None = None
_conversation_repo: PostgresConversationRepo | None = None


def get_doc_repo() -> PostgresDocumentRepo:
    global _doc_repo
    if _doc_repo is None:
        _doc_repo = PostgresDocumentRepo()
    return _doc_repo


def get_vector_repo() -> VectorStoreRepo:
    global _vector_repo
    if _vector_repo is None:
        embeddings = EmbeddingsFactory.get_embeddings()
        if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.FAISS:
            _vector_repo = FaissVectorStoreRepo(embeddings=embeddings)
        else:
            _vector_repo = PostgresVectorStoreRepo(
                embeddings=embeddings,
                enable_hybrid_search=settings.RETRIEVAL_STRATEGY == SearchStrategy.HYBRID,
            )
    return _vector_repo


def get_summary_repo() -> PostgresSummaryRepo:
    global _summary_repo
    if _summary_repo is None:
        _summary_repo = PostgresSummaryRepo()
    return _summary_repo


def get_conversation_repo() -> PostgresConversationRepo:
    global _conversation_repo
    if _conversation_repo is None:
        _conversation_repo = PostgresConversationRepo()
    return _conversation_repo


def get_summarizer() -> LangChainSummarizer:
    llm = LLMFactory.get_llm()
    return LangChainSummarizer(llm=llm)


def get_rag_service() -> LangChainRAGService:
    llm = LLMFactory.get_llm()
    return LangChainRAGService(llm=llm)


def get_reranker_service() -> RerankerService:
    return RerankerFactory.get_reranker()


def get_upload_document_use_case(
    doc_repo: Annotated[PostgresDocumentRepo, Depends(get_doc_repo)],
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
) -> UploadDocumentUseCase:
    return UploadDocumentUseCase(doc_repo, vector_repo)


def get_summarize_docs_use_case(
    doc_repo: Annotated[PostgresDocumentRepo, Depends(get_doc_repo)],
    summarizer: Annotated[LangChainSummarizer, Depends(get_summarizer)],
    summary_repo: Annotated[PostgresSummaryRepo, Depends(get_summary_repo)],
) -> SummarizeDocsUseCase:
    return SummarizeDocsUseCase(doc_repo, summarizer, summary_repo)


def get_ask_question_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    rag_service: Annotated[LangChainRAGService, Depends(get_rag_service)],
    reranker: Annotated[RerankerService, Depends(get_reranker_service)],
) -> AskQuestionUseCase:
    return AskQuestionUseCase(
        vector_repo,
        rag_service,
        reranker,
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
    )


def get_chat_with_docs_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    rag_service: Annotated[LangChainRAGService, Depends(get_rag_service)],
    conversation_repo: Annotated[PostgresConversationRepo, Depends(get_conversation_repo)],
    reranker: Annotated[RerankerService, Depends(get_reranker_service)],
) -> ChatWithDocsUseCase:
    return ChatWithDocsUseCase(
        vector_repo,
        rag_service,
        conversation_repo,
        reranker,
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
    )


def get_delete_document_use_case(
    doc_repo: Annotated[PostgresDocumentRepo, Depends(get_doc_repo)],
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
) -> DeleteDocumentUseCase:
    return DeleteDocumentUseCase(doc_repo, vector_repo)


def get_delete_summary_use_case(
    summary_repo: Annotated[PostgresSummaryRepo, Depends(get_summary_repo)],
) -> DeleteSummaryUseCase:
    return DeleteSummaryUseCase(summary_repo)


def get_list_conversations_use_case(
    conversation_repo: Annotated[PostgresConversationRepo, Depends(get_conversation_repo)],
) -> ListConversationsUseCase:
    return ListConversationsUseCase(conversation_repo)


def get_get_conversation_use_case(
    conversation_repo: Annotated[PostgresConversationRepo, Depends(get_conversation_repo)],
) -> GetConversationUseCase:
    return GetConversationUseCase(conversation_repo)


def get_delete_conversation_use_case(
    conversation_repo: Annotated[PostgresConversationRepo, Depends(get_conversation_repo)],
) -> DeleteConversationUseCase:
    return DeleteConversationUseCase(conversation_repo)
