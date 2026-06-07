from fastapi import Depends
from app.knowledge_management.domain.repositories import VectorStoreRepo
from app.knowledge_management.infrastructure.persistence.filesystem_document_repo import FilesystemDocumentRepo
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import FaissVectorStoreRepo
from app.knowledge_management.infrastructure.persistence.postgres_vectorstore_repo import PostgresVectorStoreRepo
from app.knowledge_management.infrastructure.llm.langchain_summarizer import LangChainSummarizer
from app.knowledge_management.infrastructure.llm.langchain_rag_service import LangChainRAGService
from app.knowledge_management.application.use_cases.upload_document import UploadDocumentUseCase
from app.knowledge_management.application.use_cases.summarize_docs import SummarizeDocsUseCase
from app.knowledge_management.application.use_cases.ask_question import AskQuestionUseCase
from app.knowledge_management.application.use_cases.chat_with_docs import ChatWithDocsUseCase

_doc_repo = FilesystemDocumentRepo()
_vector_repo: VectorStoreRepo | None = None
_summarizer = LangChainSummarizer()
_rag_service = LangChainRAGService()

def get_doc_repo() -> FilesystemDocumentRepo:
    return _doc_repo

def get_vector_repo() -> VectorStoreRepo:
    global _vector_repo
    if _vector_repo is None:
        try:
            _vector_repo = PostgresVectorStoreRepo()
        except Exception:
            _vector_repo = FaissVectorStoreRepo()
    return _vector_repo

def get_summarizer() -> LangChainSummarizer:
    return _summarizer

def get_rag_service() -> LangChainRAGService:
    return _rag_service

def get_upload_document_use_case(
    doc_repo: FilesystemDocumentRepo = Depends(get_doc_repo),
    vector_repo: VectorStoreRepo = Depends(get_vector_repo)
) -> UploadDocumentUseCase:
    return UploadDocumentUseCase(doc_repo, vector_repo)

def get_summarize_docs_use_case(
    doc_repo: FilesystemDocumentRepo = Depends(get_doc_repo),
    summarizer: LangChainSummarizer = Depends(get_summarizer)
) -> SummarizeDocsUseCase:
    return SummarizeDocsUseCase(doc_repo, summarizer)

def get_ask_question_use_case(
    vector_repo: VectorStoreRepo = Depends(get_vector_repo),
    rag_service: LangChainRAGService = Depends(get_rag_service)
) -> AskQuestionUseCase:
    return AskQuestionUseCase(vector_repo, rag_service)

def get_chat_with_docs_use_case(
    vector_repo: VectorStoreRepo = Depends(get_vector_repo),
    rag_service: LangChainRAGService = Depends(get_rag_service)
) -> ChatWithDocsUseCase:
    return ChatWithDocsUseCase(vector_repo, rag_service)
