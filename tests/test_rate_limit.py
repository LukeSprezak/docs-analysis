"""Testy rate limitingu (slowapi).

Limiter jest globalnie wyłączony w testach (autouse fixture w conftest), więc te testy
celowo go włączają i resetują liczniki, żeby zweryfikować, że limit faktycznie tnie.
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
    """Wyciąga liczbę dozwolonych żądań z limitu w formacie "<liczba>/<okno>"."""
    return int(limit_value.split("/")[0])


@pytest.fixture
def rate_limiting_enabled():
    """Włącza limiter na czas testu i czyści liczniki (przed i po)."""
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
        # Pierwsze `allowed` żądań przechodzi.
        for _ in range(allowed):
            response = client.post("/api/v1/auth/login", json=payload)
            assert response.status_code == 200

        # Kolejne jest odcięte limitem.
        blocked = client.post("/api/v1/auth/login", json=payload)
        assert blocked.status_code == 429
        body = blocked.json()
        assert body["error"]["error_code"] == "RATE_LIMIT_EXCEEDED"
    finally:
        app.dependency_overrides.pop(get_authenticate_user_use_case, None)


def test_rate_limit_disabled_allows_unlimited_requests():
    # Domyślnie (poza dedykowanym fixture) limiter jest wyłączony — wiele żądań przechodzi.
    assert limiter.enabled is False
    for _ in range(_parse_limit_per_window(settings.RATE_LIMIT_DEFAULT) + 5):
        response = client.get("/health")
        assert response.status_code == 200
