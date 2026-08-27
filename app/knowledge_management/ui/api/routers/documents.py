import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.knowledge_management.application.use_cases.delete_document import DeleteDocumentUseCase
from app.knowledge_management.application.use_cases.upload_document import UploadDocumentUseCase
from app.knowledge_management.domain.repositories import DocumentRepo
from app.knowledge_management.infrastructure.pdf.pymupdf_loader import PyMuPDFLoader
from app.shared.config import settings
from app.shared.dependencies import (
    get_delete_document_use_case,
    get_doc_repo,
    get_upload_document_use_case,
)
from app.shared.exceptions import ValidationException
from app.shared.rate_limit import limiter
from app.shared.storage import safe_document_path
from app.shared.upload_validation import validate_pdf_content, validate_upload_extension

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    status: str


class DocumentInfo(BaseModel):
    id: str
    filename: str


# 201: the request creates a document, the same rule `/auth/register` follows.
@router.post("/upload", response_model=DocumentResponse, status_code=201)
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    use_case: Annotated[UploadDocumentUseCase, Depends(get_upload_document_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentResponse:
    # Sanitize the name + guard against path traversal (raises 400 for "../").
    # The file goes into the user's own subdirectory (isolation on disk).
    file_path = safe_document_path(file.filename or "", current_user.id)
    filename = os.path.basename(file_path)

    # Extension allowlist — rejects an unsupported file type before anything reaches the
    # disk (instead of treating every non-PDF as text).
    extension = validate_upload_extension(filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Size limit: first by the declared size (when known), then hard by the real byte
    # count — all before anything is written to disk.
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise ValidationException(f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)")

    raw = await file.read()
    if len(raw) > max_bytes:
        raise ValidationException(f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)")

    # Content validation by magic bytes (not just by name) — before writing to disk.
    if extension == ".pdf":
        validate_pdf_content(raw)

    # Write to disk in a threadpool so the event loop is not blocked (ASYNC230).
    await run_in_threadpool(Path(file_path).write_bytes, raw)

    pages = None
    if extension == ".pdf":
        try:
            loader = PyMuPDFLoader()
            # Parsing is synchronous and CPU-bound (PyMuPDF's C extension) — in a threadpool
            # for the same reason as the write above, and a more pressing one: a large PDF
            # takes seconds, during which the process would answer nothing at all, /health
            # included.
            pages = await run_in_threadpool(loader.load_pdf, file_path)
            if not pages:
                raise ValidationException("Could not extract text from PDF")
            # The full content (for listing/summaries); splitting pages for retrieval
            # happens in the vector layer.
            text = "\n\n".join(page.content for page in pages)
        except ValidationException:
            raise
        except Exception as e:
            # The exception detail goes to the logs only — the client gets a generic message
            # (no internal information leaking through the error text).
            logger.exception("Failed to process PDF: %s", filename)
            raise ValidationException("The uploaded PDF could not be processed") from e
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValidationException("File is not a valid UTF-8 text file") from e

    # Unexpected errors (e.g. from the use case) propagate to global_exception_handler,
    # which logs the detail and returns a generic 500 — no `str(e)` reaches the client.
    document = await use_case.execute(
        doc_id=filename,
        content=text,
        metadata={"filename": filename, "file_path": file_path},
        owner_id=current_user.id,
        pages=pages,
    )
    return DocumentResponse(id=document.id, status="uploaded")


@router.get("/", response_model=list[DocumentInfo])
async def list_documents(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=settings.LIST_MAX_LIMIT)] = settings.LIST_DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentInfo]:
    documents = await doc_repo.list_all(current_user.id, limit=limit, offset=offset)
    return [
        DocumentInfo(id=doc.id, filename=doc.metadata.get("filename", doc.id)) for doc in documents
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    use_case: Annotated[DeleteDocumentUseCase, Depends(get_delete_document_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    await use_case.execute(document_id, current_user.id)
    return {"status": "success"}
