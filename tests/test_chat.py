import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.knowledge_management.domain.models import Answer, Conversation, Document
from app.main import app
from app.shared.dependencies import get_chat_with_docs_use_case, get_list_conversations_use_case

client = TestClient(app, raise_server_exceptions=False)


def test_chat_stream_endpoint_emits_ndjson_tokens_then_done():
    mock_use_case = MagicMock()

    async def fake_stream(message, owner_id, history, conversation_id):
        yield {"type": "token", "content": "Hel"}
        yield {"type": "token", "content": "lo"}
        yield {
            "type": "done",
            "conversation_id": "c1",
            "sources": [
                Document(id="doc1", content="x", metadata={"filename": "doc1.pdf", "page": 2})
            ],
        }

    mock_use_case.execute_stream = fake_stream
    app.dependency_overrides[get_chat_with_docs_use_case] = lambda: mock_use_case

    response = client.post("/api/v1/chat/stream", json={"message": "hi"})

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0] == {"type": "token", "content": "Hel"}
    assert events[1] == {"type": "token", "content": "lo"}
    assert events[-1]["type"] == "done"
    assert events[-1]["conversation_id"] == "c1"
    # The sources are formatted using format_sources (title + page)
    assert events[-1]["sources"] == ["doc1.pdf (page 2)"]

    app.dependency_overrides.clear()


def test_chat_endpoint():
    # Mock result
    mock_result = Answer(
        text="The answer is 42",
        sources=[Document(id="doc1", content="meaning of life", metadata={})],
    )

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=(mock_result, "conv123"))

    # Override dependency
    app.dependency_overrides[get_chat_with_docs_use_case] = lambda: mock_use_case

    response = client.post("/api/v1/chat/", json={"message": "What is the meaning of life?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer is 42"
    assert data["conversation_id"] == "conv123"
    assert "doc1" in data["sources"]

    # Cleanup
    app.dependency_overrides.clear()


def test_list_conversations():
    mock_conversations = [
        Conversation(id="c1", title="Conv 1", messages=[]),
        Conversation(id="c2", title="Conv 2", messages=[]),
    ]

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=mock_conversations)

    app.dependency_overrides[get_list_conversations_use_case] = lambda: mock_use_case

    response = client.get("/api/v1/chat/conversations")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == "c1"
    assert data[1]["id"] == "c2"

    app.dependency_overrides.clear()
