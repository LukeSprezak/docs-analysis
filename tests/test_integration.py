from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.knowledge_management.domain.models import Document
from app.main import app
from app.shared.dependencies import get_doc_repo, get_upload_document_use_case

client = TestClient(app)


def test_list_documents_empty(override_dependency):
    mock_repo = MagicMock()
    mock_repo.list_all = AsyncMock(return_value=[])
    override_dependency(get_doc_repo, lambda: mock_repo)

    response = client.get("/api/v1/documents/")
    assert response.status_code == 200
    assert response.json() == []


def test_upload_and_list_documents(override_dependency):
    # Mock upload use case
    mock_upload = MagicMock()
    mock_upload.execute = AsyncMock(
        return_value=Document(
            id="test.txt", content="hello world", metadata={"filename": "test.txt"}
        )
    )
    override_dependency(get_upload_document_use_case, lambda: mock_upload)

    # Mock list repo
    mock_repo = MagicMock()
    mock_repo.list_all = AsyncMock(
        return_value=[
            Document(id="test.txt", content="hello world", metadata={"filename": "test.txt"})
        ]
    )
    override_dependency(get_doc_repo, lambda: mock_repo)

    # Upload
    files = {"file": ("test.txt", b"hello world")}
    upload_resp = client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code == 201

    # List
    list_resp = client.get("/api/v1/documents/")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert len(data) > 0
    assert any(doc["filename"] == "test.txt" for doc in data)


def test_cors_headers():
    response = client.options(
        "/api/v1/documents/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
