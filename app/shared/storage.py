import os

from app.shared.config import settings
from app.shared.exceptions import ValidationException


def storage_documents_dir() -> str:
    """Absolute path of the documents root, resolved on every call.

    Deliberately not a module constant: a relative setting resolves against the process cwd,
    and tests repoint it at a temporary directory — a value frozen at import time would miss
    both.
    """
    return os.path.abspath(settings.STORAGE_DOCUMENTS_DIR)


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

    root = storage_documents_dir()
    path = os.path.abspath(os.path.join(root, owner_id, name))

    if os.path.commonpath([root, path]) != root:
        raise ValidationException("Invalid filename")
    return path


def is_within_storage(path: str) -> bool:
    """Whether the path lies inside the storage directory (used for safe deletion)."""
    root = storage_documents_dir()
    try:
        resolved = os.path.abspath(path)
        return os.path.commonpath([root, resolved]) == root
    except ValueError:
        return False
