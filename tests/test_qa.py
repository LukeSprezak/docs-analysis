from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.knowledge_management.domain.models import Answer, Document
from app.main import app
from app.shared.dependencies import get_ask_question_use_case

client = TestClient(app, raise_server_exceptions=False)


def test_ask_question_endpoint(override_dependency):
    mock_result = Answer(
        text="FastAPI is a modern web framework",
        sources=[Document(id="doc2", content="FastAPI docs", metadata={})],
    )

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=mock_result)

    override_dependency(get_ask_question_use_case, lambda: mock_use_case)

    response = client.post("/api/v1/qa/ask", json={"question": "What is FastAPI?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FastAPI is a modern web framework"
    assert "doc2" in data["sources"]
