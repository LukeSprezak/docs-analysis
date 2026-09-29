"""The retrieval context's public surface: other contexts import from here and nowhere else."""

from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase
from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.dependencies import (
    get_index_document_use_case,
    get_rag_service,
    get_remove_from_index_use_case,
    get_retrieval_pipeline,
)
from app.retrieval.domain.models import Answer
from app.retrieval.domain.repositories import RAGService
from app.retrieval.ui.sources import format_sources

__all__ = [
    "Answer",
    "IndexDocumentUseCase",
    "RAGService",
    "RemoveFromIndexUseCase",
    "RetrievalPipeline",
    "format_sources",
    "get_index_document_use_case",
    "get_rag_service",
    "get_remove_from_index_use_case",
    "get_retrieval_pipeline",
]
