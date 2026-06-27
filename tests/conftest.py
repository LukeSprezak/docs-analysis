import pytest

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.main import app
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