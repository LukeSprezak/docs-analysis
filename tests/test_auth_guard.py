from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.identity.dependencies import get_user_repo
from app.identity.models import User
from app.identity.security import create_access_token
from app.main import app
from app.retrieval.dependencies import get_rag_service, get_retrieval_pipeline
from app.shared.kernel.document import Document

client = TestClient(app, raise_server_exceptions=False)


async def get_by_email():  # pragma: no cover -  not used in the test
    return None


class FakeUserRepo:
    def __init__(self, user: User | None = None):
        self.user = user

    async def get_by_id(self, user_id: str) -> User | None:
        if self.user and self.user.id == user_id:
            return self.user
        return None

    async def save(self, user):  # pragma: no cover
        pass


def test_protected_endpoint_without_token_returns_401(without_test_user, override_dependency):
    override_dependency(get_user_repo, lambda: FakeUserRepo())
    override_dependency(get_retrieval_pipeline, lambda: MagicMock())
    override_dependency(get_rag_service, lambda: MagicMock())

    response = client.post("/api/v1/qa/ask", json={"question": "anything"})

    assert response.status_code == 401


def test_protected_endpoint_with_invalid_token_returns_401(without_test_user, override_dependency):
    override_dependency(get_user_repo, lambda: FakeUserRepo())
    override_dependency(get_retrieval_pipeline, lambda: MagicMock())
    override_dependency(get_rag_service, lambda: MagicMock())

    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "anything"},
        headers={"Authorization": "Bearer not-a-token"},
    )

    assert response.status_code == 401


def test_valid_token_passes_guard_and_reaches_retrieval(without_test_user, override_dependency):
    user = User(id="u1", email="alice@example.com", hashed_password="x")
    override_dependency(get_user_repo, lambda: FakeUserRepo(user))

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve = AsyncMock(
        return_value=[Document(id="u1::doc.pdf", content="x", metadata={"filename": "doc.pdf"})]
    )
    mock_rag_service = MagicMock()
    mock_rag_service.answer_question = AsyncMock(return_value="answer")
    override_dependency(get_retrieval_pipeline, lambda: mock_pipeline)
    override_dependency(get_rag_service, lambda: mock_rag_service)

    token = create_access_token(user.id)
    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "question"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "answer"
    # retrieval received the logged-in user's ID as the owner_id
    assert mock_pipeline.retrieve.call_args.args[1] == "u1"
