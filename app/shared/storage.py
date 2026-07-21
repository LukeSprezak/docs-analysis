import os

from app.shared.exceptions import ValidationException

# Katalog na wgrane dokumenty. Liczony od cwd (aplikacja startuje z roota projektu),
# zgodnie z dotychczasowym użyciem "storage/documents".
STORAGE_DOCUMENTS_DIR = os.path.abspath("storage/documents")


def safe_document_path(filename: str, owner_id: str) -> str:
    """Zwraca bezpieczną, absolutną ścieżkę pliku wewnątrz katalogu storage użytkownika.

    Chroni przed path traversal: bierze samą nazwę pliku (`basename`), odrzuca
    puste/`.`/`..` i upewnia się, że wynik nie wychodzi poza katalog storage.
    Pliki każdego użytkownika trafiają do podkatalogu `owner_id`, więc dwóch userów
    może wgrać plik o tej samej nazwie bez kolizji na dysku. Podnosi
    ValidationException (400) dla niepoprawnych nazw.
    """
    raw = filename or ""
    name = os.path.basename(raw)
    # Fail-closed: odrzucamy nazwy ze składnikami ścieżki zamiast je po cichu
    # przycinać — jawny kontrakt zamiast niejawnego zmieniania nazwy pliku.
    if not name or name in (".", "..") or name != raw or "/" in raw or "\\" in raw:
        raise ValidationException("Invalid filename")

    # owner_id jest generowany po stronie serwera (UUID), ale defensywnie odrzucamy
    # wszystko, co mogłoby wyjść z katalogu (separatory / "..").
    if not owner_id or owner_id != os.path.basename(owner_id) or owner_id in (".", ".."):
        raise ValidationException("Invalid owner")

    path = os.path.abspath(os.path.join(STORAGE_DOCUMENTS_DIR, owner_id, name))
    if os.path.commonpath([STORAGE_DOCUMENTS_DIR, path]) != STORAGE_DOCUMENTS_DIR:
        raise ValidationException("Invalid filename")
    return path


def is_within_storage(path: str) -> bool:
    """Czy ścieżka leży wewnątrz katalogu storage (do bezpiecznego usuwania)."""
    try:
        resolved = os.path.abspath(path)
        return os.path.commonpath([STORAGE_DOCUMENTS_DIR, resolved]) == STORAGE_DOCUMENTS_DIR
    except ValueError:
        # commonpath rzuca przy ścieżkach na różnych dyskach (Windows) itp.
        return False
