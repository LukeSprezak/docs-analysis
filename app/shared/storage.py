import os

from app.shared.exceptions import ValidationException

STORAGE_DOCUMENTS_DIR = os.path.abspath("storage/documents")


def safe_document_path(filename: str, owner_id: str) -> str:
    """Returns a safe, absolute file path inside the user's storage directory.

    Protects against path traversal: takes the bare file name (`basename`), rejects
    empty/`.`/`..` and makes sure the result does not escape the storage directory.
    Each user's files go into an `owner_id` subdirectory, so two users can upload a file
    with the same name without colliding on disk. Raises ValidationException (400) for
    invalid names.
    """
    raw = filename or ""
    name = os.path.basename(raw)

    if not name or name in (".", "..") or name != raw or "/" in raw or "\\" in raw:
        raise ValidationException("Invalid filename")

    if not owner_id or owner_id != os.path.basename(owner_id) or owner_id in (".", ".."):
        raise ValidationException("Invalid owner")

    path = os.path.abspath(os.path.join(STORAGE_DOCUMENTS_DIR, owner_id, name))

    if os.path.commonpath([STORAGE_DOCUMENTS_DIR, path]) != STORAGE_DOCUMENTS_DIR:
        raise ValidationException("Invalid filename")
    return path


def is_within_storage(path: str) -> bool:
    """Whether the path lies inside the storage directory (used for safe deletion)."""
    try:
        resolved = os.path.abspath(path)
        return os.path.commonpath([STORAGE_DOCUMENTS_DIR, resolved]) == STORAGE_DOCUMENTS_DIR
    except ValueError:
        return False
