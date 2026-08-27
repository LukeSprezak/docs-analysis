import json
import uuid
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.knowledge_management.domain.models import Answer, Conversation, Document
from app.main import app
from app.shared.dependencies import get_chat_with_docs_use_case, get_list_conversations_use_case
from app.shared.exceptions import EntityNotFoundException

client = TestClient(app, raise_server_exceptions=False)


def test_chat_stream_endpoint_emits_ndjson_tokens_then_done(override_dependency):
    mock_use_case = MagicMock()

    async def fake_stream(message, owner_id, conversation_id):
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
    override_dependency(get_chat_with_docs_use_case, lambda: mock_use_case)

    response = client.post("/api/v1/chat/stream", json={"message": "hi"})

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0] == {"type": "token", "content": "Hel"}
    assert events[1] == {"type": "token", "content": "lo"}
    assert events[-1]["type"] == "done"
    assert events[-1]["conversation_id"] == "c1"
    # The sources are formatted using format_sources (title + page)
    assert events[-1]["sources"] == ["doc1.pdf (page 2)"]


def test_chat_stream_endpoint_reports_an_unknown_conversation_as_404(override_dependency):
    """SEC-01, streaming half: the rejection must reach the client as a status code.

    The use case refuses a conversation id the caller does not own. If that refusal only
    surfaces once StreamingResponse is already iterating, the 200 headers are on the wire
    and the client sees a truncated stream instead of an error.
    """
    mock_use_case = MagicMock()

    async def fake_stream(message, owner_id, conversation_id):
        raise EntityNotFoundException(entity="Conversation", identifier=conversation_id)
        yield  # pragma: no cover - keeps this an async generator

    mock_use_case.execute_stream = fake_stream
    override_dependency(get_chat_with_docs_use_case, lambda: mock_use_case)

    response = client.post(
        "/api/v1/chat/stream",
        json={"message": "hi", "conversation_id": str(uuid.uuid4())},
    )

    assert response.status_code == 404


def test_chat_stream_endpoint_rejects_a_malformed_conversation_id_as_422():
    """CON-05: a malformed id is a client error, not a database cast error.

    `conversations.id` is a `uuid` column, so an id that is not one reaches Postgres and comes
    back as a 500 — a plain typo in a URL would page whoever watches the error rate."""
    response = client.post(
        "/api/v1/chat/stream", json={"message": "hi", "conversation_id": "not-a-uuid"}
    )

    assert response.status_code == 422


def test_chat_endpoint(override_dependency):
    # Mock result
    mock_result = Answer(
        text="The answer is 42",
        sources=[Document(id="doc1", content="meaning of life", metadata={})],
    )

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=(mock_result, "conv123"))

    # Override dependency
    override_dependency(get_chat_with_docs_use_case, lambda: mock_use_case)

    response = client.post("/api/v1/chat/", json={"message": "What is the meaning of life?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer is 42"
    assert data["conversation_id"] == "conv123"
    assert "doc1" in data["sources"]

    # Cleanup


def test_list_conversations(override_dependency):
    mock_conversations = [
        Conversation(id="c1", title="Conv 1", messages=[]),
        Conversation(id="c2", title="Conv 2", messages=[]),
    ]

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=mock_conversations)

    override_dependency(get_list_conversations_use_case, lambda: mock_use_case)

    response = client.get("/api/v1/chat/conversations")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == "c1"
    assert data[1]["id"] == "c2"
