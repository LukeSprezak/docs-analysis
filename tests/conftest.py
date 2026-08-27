import pytest

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.main import app
from app.shared.config import settings
from app.shared.rate_limit import limiter

TEST_USER = User(id="test-user-id", email="test@example.com", hashed_password="x")


@pytest.fixture(autouse=True)
def authenticated_test_user():
    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """The limiter is disabled during testing—repeated requests from the same address
    (TestClient) would hit the limit. The test designed to check rate limiting intentionally
    enables it (limiter.enabled = True) and resets the counters."""
    limiter.enabled = False
    limiter.reset()
    yield
    limiter.enabled = False
    limiter.reset()


@pytest.fixture(autouse=True)
def storage_in_tmp_path(tmp_path, monkeypatch):
    """Uploads land in this test's own directory, not in the project's `storage/`.

    Without it every upload test writes into the working tree, leaving files behind that the
    next run reads back — and two runs in parallel would fight over the same names."""
    monkeypatch.setattr(settings, "STORAGE_DOCUMENTS_DIR", str(tmp_path / "documents"))
