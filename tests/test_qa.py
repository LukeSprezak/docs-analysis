from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.dependencies import get_rag_service, get_retrieval_pipeline
from app.shared.kernel.document import Document

client = TestClient(app, raise_server_exceptions=False)


def test_ask_question_endpoint(override_dependency):
    sources = [Document(id="doc2", content="FastAPI docs", metadata={})]
    mock_pipeline = MagicMock()
    mock_pipeline.retrieve = AsyncMock(return_value=sources)
    mock_rag_service = MagicMock()
    mock_rag_service.answer_question = AsyncMock(return_value="FastAPI is a modern web framework")

    override_dependency(get_retrieval_pipeline, lambda: mock_pipeline)
    override_dependency(get_rag_service, lambda: mock_rag_service)

    response = client.post("/api/v1/qa/ask", json={"question": "What is FastAPI?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FastAPI is a modern web framework"
    assert "doc2" in data["sources"]
    # The RAG receives exactly the retrieved fragments.
    assert mock_rag_service.answer_question.call_args.args[1] == sources
