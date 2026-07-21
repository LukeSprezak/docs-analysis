"""Rate limiting tests (slowapi).

The limiter is globally disabled in the tests (autouse fixture in conftest), so these tests
deliberately enable it and reset the counters to verify that the limit really does cut requests off.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.identity.dependencies import get_authenticate_user_use_case
from app.identity.domain.models import User
from app.main import app
from app.shared.config import settings
from app.shared.rate_limit import limiter

client = TestClient(app, raise_server_exceptions=False)


def _parse_limit_per_window(limit_value: str) -> int:
    """Extracts the number of allowed requests from a limit in the "<count>/<window>" format."""
    return int(limit_value.split("/")[0])


@pytest.fixture
def rate_limiting_enabled():
    """Enables the limiter for the duration of the test and clears the counters (before and after)."""
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.enabled = False
    limiter.reset()


def test_auth_login_blocks_after_limit(rate_limiting_enabled):
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(
        return_value=User(id="u1", email="user@example.com", hashed_password="x")
    )
    app.dependency_overrides[get_authenticate_user_use_case] = lambda: mock_use_case

    allowed = _parse_limit_per_window(settings.RATE_LIMIT_AUTH)
    payload = {"email": "user@example.com", "password": "secret"}

    try:
        # The first `allowed` requests go through.
        for _ in range(allowed):
            response = client.post("/api/v1/auth/login", json=payload)
            assert response.status_code == 200

        # The next one is cut off by the limit.
        blocked = client.post("/api/v1/auth/login", json=payload)
        assert blocked.status_code == 429
        body = blocked.json()
        assert body["error"]["error_code"] == "RATE_LIMIT_EXCEEDED"
    finally:
        app.dependency_overrides.pop(get_authenticate_user_use_case, None)


def test_rate_limit_disabled_allows_unlimited_requests():
    # By default (outside the dedicated fixture) the limiter is off — many requests go through.
    assert limiter.enabled is False
    for _ in range(_parse_limit_per_window(settings.RATE_LIMIT_DEFAULT) + 5):
        response = client.get("/health")
        assert response.status_code == 200
