import os

from app.shared.config import settings
from app.shared.exceptions import ValidationException

# Nagłówek pliku PDF wg specyfikacji ("%PDF-1.x"). Sprawdzamy go zamiast ufać samemu
# rozszerzeniu czy deklarowanemu przez klienta content-type.
PDF_MAGIC_BYTES = b"%PDF-"

# W ilu pierwszych bajtach szukamy nagłówka PDF. Część plików ma drobny śmieć/whitespace
# przed nagłówkiem (tolerowany przez czytniki) — dajemy mu margines, ale skończony.
PDF_HEADER_SCAN_BYTES = 1024


def allowed_upload_extensions() -> set[str]:
    """Dozwolone rozszerzenia uploadu z configu, znormalizowane do małych liter z kropką."""
    return {
        extension.strip().lower()
        for extension in settings.ALLOWED_UPLOAD_EXTENSIONS.split(",")
        if extension.strip()
    }


def validate_upload_extension(filename: str) -> str:
    """Sprawdza, że rozszerzenie pliku jest na allowliście; zwraca je (małe litery).

    Pozwala odrzucić niewspierany typ pliku zanim cokolwiek trafi na dysk, zamiast
    traktować dowolny nie-PDF jako tekst. Podnosi ValidationException (400).
    """
    extension = os.path.splitext(filename)[1].lower()
    allowed = allowed_upload_extensions()
    if extension not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationException(f"Unsupported file type. Allowed: {allowed_list}")
    return extension


def validate_pdf_content(raw_bytes: bytes) -> None:
    """Weryfikuje, że treść faktycznie jest PDF-em (nagłówek `%PDF-`), nie tylko po nazwie.

    Chroni przed plikiem podszywającym się pod PDF (np. wykonywalny zapisany jako
    `.pdf`). Podnosi ValidationException (400), gdy nagłówka brak.
    """
    if PDF_MAGIC_BYTES not in raw_bytes[:PDF_HEADER_SCAN_BYTES]:
        raise ValidationException("File content is not a valid PDF")
