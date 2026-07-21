import os

from app.shared.config import settings
from app.shared.exceptions import ValidationException

# The PDF file header per the specification ("%PDF-1.x"). We check it instead of trusting
# the extension alone or the content-type declared by the client.
PDF_MAGIC_BYTES = b"%PDF-"

# How many leading bytes we scan for the PDF header. Some files carry a little junk or
# whitespace before it (readers tolerate that) — we allow a margin, but a bounded one.
PDF_HEADER_SCAN_BYTES = 1024


def allowed_upload_extensions() -> set[str]:
    """Allowed upload extensions from config, normalized to lowercase with a leading dot."""
    return {
        extension.strip().lower()
        for extension in settings.ALLOWED_UPLOAD_EXTENSIONS.split(",")
        if extension.strip()
    }


def validate_upload_extension(filename: str) -> str:
    """Checks that the file extension is on the allowlist; returns it (lowercased).

    Lets us reject an unsupported file type before anything reaches the disk, instead of
    treating every non-PDF as text. Raises ValidationException (400).
    """
    extension = os.path.splitext(filename)[1].lower()
    allowed = allowed_upload_extensions()
    if extension not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationException(f"Unsupported file type. Allowed: {allowed_list}")
    return extension


def validate_pdf_content(raw_bytes: bytes) -> None:
    """Verifies the content really is a PDF (the `%PDF-` header), not just by its name.

    Protects against a file impersonating a PDF (e.g. an executable saved as `.pdf`).
    Raises ValidationException (400) when the header is missing.
    """
    if PDF_MAGIC_BYTES not in raw_bytes[:PDF_HEADER_SCAN_BYTES]:
        raise ValidationException("File content is not a valid PDF")
