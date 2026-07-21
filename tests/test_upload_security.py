import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.knowledge_management.domain.models import Document
from app.main import app
from app.shared.dependencies import get_upload_document_use_case
from app.shared.exceptions import ValidationException
from app.shared.storage import STORAGE_DOCUMENTS_DIR, is_within_storage, safe_document_path
from app.shared.upload_validation import validate_pdf_content, validate_upload_extension

client = TestClient(app, raise_server_exceptions=False)


def _mock_upload():
    mock = MagicMock()
    mock.execute.return_value = Document(id="ok.txt", content="x", metadata={"filename": "ok.txt"})
    return mock


@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "../secret.txt", "some/dir/file.txt", "a\\b.txt", "..", ".", ""],
)
def test_safe_document_path_rejects_path_components(bad):
    with pytest.raises(ValidationException):
        safe_document_path(bad, "owner-1")


def test_safe_document_path_accepts_plain_filename():
    path = safe_document_path("file.txt", "owner-1")
    assert os.path.basename(path) == "file.txt"
    # The file lands in the owner's subdirectory, but still inside the storage.
    assert os.path.basename(os.path.dirname(path)) == "owner-1"
    assert is_within_storage(path)


def test_safe_document_path_rejects_malicious_owner():
    with pytest.raises(ValidationException):
        safe_document_path("file.txt", "../escape")


def test_is_within_storage():
    assert is_within_storage(os.path.join(STORAGE_DOCUMENTS_DIR, "a.txt"))
    assert not is_within_storage("/etc/passwd")


def test_upload_rejects_path_traversal_filename():
    app.dependency_overrides[get_upload_document_use_case] = _mock_upload
    files = {"file": ("../../evil.txt", b"pwned")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 400
    app.dependency_overrides.clear()


def test_upload_rejects_oversized_file():
    app.dependency_overrides[get_upload_document_use_case] = _mock_upload
    from app.shared.config import settings

    too_big = b"a" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    files = {"file": ("big.txt", too_big)}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 400
    app.dependency_overrides.clear()


@pytest.mark.parametrize("good", ["file.pdf", "FILE.PDF", "note.txt", "readme.md"])
def test_validate_upload_extension_accepts_allowlisted(good):
    assert validate_upload_extension(good) == os.path.splitext(good)[1].lower()


@pytest.mark.parametrize("bad", ["evil.exe", "archive.zip", "script.sh", "image.png", "noext"])
def test_validate_upload_extension_rejects_others(bad):
    with pytest.raises(ValidationException):
        validate_upload_extension(bad)


def test_validate_pdf_content_accepts_real_header():
    validate_pdf_content(b"%PDF-1.7\n...rest of the file...")


@pytest.mark.parametrize("bad", [b"not a pdf at all", b"MZ\x90\x00", b"", b"PK\x03\x04zip"])
def test_validate_pdf_content_rejects_non_pdf(bad):
    with pytest.raises(ValidationException):
        validate_pdf_content(bad)


def test_upload_rejects_disallowed_extension():
    app.dependency_overrides[get_upload_document_use_case] = _mock_upload
    files = {"file": ("evil.exe", b"MZ\x90\x00binary")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 400
    app.dependency_overrides.clear()


def test_upload_rejects_pdf_with_wrong_magic_bytes():
    # A file with a .pdf extension but non-PDF content → rejected on magic bytes.
    app.dependency_overrides[get_upload_document_use_case] = _mock_upload
    files = {"file": ("fake.pdf", b"this is plain text pretending to be a pdf")}
    resp = client.post("/api/v1/documents/upload", files=files)
    assert resp.status_code == 400
    app.dependency_overrides.clear()


def test_upload_does_not_leak_exception_detail_to_client():
    # An unexpected use case error carrying sensitive text must not reach the response.
    secret_detail = "secret connection string postgres://user:pass@host"
    mock = MagicMock()
    mock.execute = AsyncMock(side_effect=RuntimeError(secret_detail))
    app.dependency_overrides[get_upload_document_use_case] = lambda: mock

    files = {"file": ("ok.txt", b"hello world")}
    resp = client.post("/api/v1/documents/upload", files=files)

    assert resp.status_code == 500
    assert secret_detail not in resp.text
    assert resp.json()["error"]["message"] == "An unexpected error occurred"
    app.dependency_overrides.clear()