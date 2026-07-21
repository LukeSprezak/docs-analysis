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
from app.knowledge_management.infrastructure.pdf.pymupdf_loader import PyMuPDFLoader
from app.knowledge_management.infrastructure.persistence.postgres_document_repo import (
    PostgresDocumentRepo,
)
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


@router.post("/upload", response_model=DocumentResponse)
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    use_case: Annotated[UploadDocumentUseCase, Depends(get_upload_document_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentResponse:
    # Sanityzacja nazwy + ochrona przed path traversal (rzuca 400 dla "../").
    # Plik trafia do podkatalogu użytkownika (izolacja na dysku).
    file_path = safe_document_path(file.filename or "", current_user.id)
    filename = os.path.basename(file_path)

    # Allowlista rozszerzeń — odrzuca niewspierany typ pliku zanim cokolwiek trafi na
    # dysk (zamiast traktować dowolny nie-PDF jako tekst).
    extension = validate_upload_extension(filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Limit rozmiaru: najpierw po deklarowanym rozmiarze (jeśli znany), potem
    # twardo po realnej liczbie bajtów — zanim cokolwiek zapiszemy na dysk.
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise ValidationException(f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)")

    raw = await file.read()
    if len(raw) > max_bytes:
        raise ValidationException(f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)")

    # Walidacja treści po magic bytes (nie tylko po nazwie) — przed zapisem na dysk.
    if extension == ".pdf":
        validate_pdf_content(raw)

    # Zapis na dysk w threadpoolu, żeby nie blokować event loopu (ASYNC230).
    await run_in_threadpool(Path(file_path).write_bytes, raw)

    pages = None
    if extension == ".pdf":
        try:
            loader = PyMuPDFLoader()
            pages = loader.load_pdf(file_path)
            if not pages:
                raise ValidationException("Could not extract text from PDF")
            # Pełna treść (do listy/podsumowań); fragmentacja stron pod
            # retrieval dzieje się w warstwie wektorowej.
            text = "\n\n".join(page.content for page in pages)
        except ValidationException:
            raise
        except Exception as e:
            # Szczegół wyjątku tylko do logów — klient dostaje ogólny komunikat
            # (bez wycieku wewnętrznych informacji przez treść błędu).
            logger.exception("Nie udało się przetworzyć PDF-a: %s", filename)
            raise ValidationException("The uploaded PDF could not be processed") from e
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValidationException("File is not a valid UTF-8 text file") from e

    # Nieoczekiwane błędy (np. z use case) propagują do global_exception_handler,
    # który loguje szczegół i zwraca ogólny 500 — żadnego `str(e)` do klienta.
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
    doc_repo: Annotated[PostgresDocumentRepo, Depends(get_doc_repo)],
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
