"""Tests for document list pagination — limit/offset passthrough + bounds validation.

The repo is replaced with a mock (the SQL LIMIT/OFFSET logic needs a live database), so what we
check is the HTTP-layer contract: that the parameters reach the repo and that FastAPI rejects
out-of-range values.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.shared.config import settings
from app.shared.dependencies import get_doc_repo
from tests.conftest import InstallOverride

client = TestClient(app, raise_server_exceptions=False)


def _mock_doc_repo() -> MagicMock:
    repo = MagicMock()
    repo.list_all = AsyncMock(return_value=[])
    return repo


def test_list_documents_passes_limit_and_offset_to_repo(
    override_dependency: InstallOverride,
) -> None:
    repo = _mock_doc_repo()
    override_dependency(get_doc_repo, lambda: repo)

    resp = client.get("/api/v1/documents/?limit=10&offset=5")

    assert resp.status_code == 200
    assert repo.list_all.call_args.kwargs == {"limit": 10, "offset": 5}


def test_list_documents_uses_config_default_limit(override_dependency: InstallOverride) -> None:
    repo = _mock_doc_repo()
    override_dependency(get_doc_repo, lambda: repo)

    resp = client.get("/api/v1/documents/")

    assert resp.status_code == 200
    assert repo.list_all.call_args.kwargs == {
        "limit": settings.LIST_DEFAULT_LIMIT,
        "offset": 0,
    }


def test_list_documents_rejects_out_of_range_pagination(
    override_dependency: InstallOverride,
) -> None:
    repo = _mock_doc_repo()
    override_dependency(get_doc_repo, lambda: repo)

    assert client.get("/api/v1/documents/?limit=0").status_code == 422
    assert client.get(f"/api/v1/documents/?limit={settings.LIST_MAX_LIMIT + 1}").status_code == 422
    assert client.get("/api/v1/documents/?offset=-1").status_code == 422
