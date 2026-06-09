import os

from app.shared.config import settings
from app.shared.exceptions import ValidationException

PDF_MAGIC_BYTES = b"%PDF-"

PDF_HEADER_SCAN_BYTES = 1024


def allowed_upload_extensions() -> set[str]:
    return {
        extension.strip().lower()
        for extension in settings.ALLOWED_UPLOAD_EXTENSIONS.split(",")
        if extension.strip()
    }


def validate_upload_extension(filename: str) -> str:
    extension = os.path.splitext(filename)[1].lower()
    allowed = allowed_upload_extensions()
    if extension not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationException(f"Unsupported file type. Allowed: {allowed_list}")
    return extension


def validate_pdf_content(raw_bytes: bytes) -> None:
    if PDF_MAGIC_BYTES not in raw_bytes[:PDF_HEADER_SCAN_BYTES]:
        raise ValidationException("File content is not a valid PDF")
