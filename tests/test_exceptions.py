from fastapi.testclient import TestClient

from app.main import app
from app.shared.exceptions import AppException, EntityNotFoundException

client = TestClient(app, raise_server_exceptions=False)


def test_app_exception_handler():
    @app.get("/test-exception")
    async def test_route():
        raise AppException(message="Test error", status_code=400, error_code="TEST_ERROR", context={"foo": "bar"})

    response = client.get("/test-exception")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error"
    assert data["error"]["error_code"] == "TEST_ERROR"
    assert data["error"]["context"] == {"foo": "bar"}
    assert "request_id" in data["error"]


def test_global_exception_handler():
    @app.get("/test-unhandled-exception")
    async def test_route():
        raise ValueError("Something went wrong")

    response = client.get("/test-unhandled-exception")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "An unexpected error occurred"
    assert data["error"]["error_code"] == "INTERNAL_SERVER_ERROR"


def test_entity_not_found_exception():
    @app.get("/test-not-found")
    async def test_route():
        raise EntityNotFoundException(entity="User", identifier=123)

    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["error_code"] == "ENTITY_NOT_FOUND"
    assert "User" in data["error"]["message"]
    assert "123" in data["error"]["message"]
    assert data["error"]["context"] == {"entity": "User", "identifier": 123}
