"""The retrieval context's public surface: other contexts import from here and nowhere else."""

from app.retrieval.dependencies import (
    get_index_document_use_case,
    get_rag_service,
    get_remove_from_index_use_case,
    get_retrieval_pipeline,
)
from app.retrieval.indexing import IndexDocumentUseCase, RemoveFromIndexUseCase
from app.retrieval.models import Answer
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.ports import RAGService
from app.retrieval.sources import format_sources

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
