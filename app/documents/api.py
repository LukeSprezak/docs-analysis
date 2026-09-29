"""The documents context's public surface: other contexts import from here and nowhere else."""

from app.documents.dependencies import get_doc_repo
from app.documents.domain.repositories import DocumentRepo

__all__ = ["DocumentRepo", "get_doc_repo"]
