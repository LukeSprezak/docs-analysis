from fastapi.testclient import TestClient

from app.main import app
from app.shared.config import settings

client = TestClient(app)

ALLOWED_ORIGIN = "http://localhost:3000"
DISALLOWED_ORIGIN = "http://evil.example.com"


def test_allowed_origin_is_echoed_not_wildcard():
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert response.status_code == 200
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin == ALLOWED_ORIGIN
    assert allow_origin != "*"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_disallowed_origin_gets_no_cors_header():
    response = client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})
    assert response.status_code == 200
    # No origin = the browser will block cross-origin requests (there is also no “*”).
    assert response.headers.get("access-control-allow-origin") != DISALLOWED_ORIGIN
    assert response.headers.get("access-control-allow-origin") != "*"


def test_preflight_allows_configured_origin():
    response = client.options(
        "/api/v1/qa/ask",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


def test_default_allowed_origins_do_not_contain_wildcard():
    assert "*" not in settings.cors_allowed_origins_list
    assert ALLOWED_ORIGIN in settings.cors_allowed_origins_list
