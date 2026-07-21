from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.identity.dependencies import get_current_user, get_user_repo
from app.identity.domain.models import User
from app.identity.security import create_access_token
from app.knowledge_management.domain.models import Answer, Document
from app.main import app
from app.shared.dependencies import get_ask_question_use_case

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


def test_protected_endpoint_without_token_returns_401():
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_user_repo] = lambda: FakeUserRepo()
    app.dependency_overrides[get_ask_question_use_case] = lambda: MagicMock()

    response = client.post("/api/v1/qa/ask", json={"question": "anything"})

    assert response.status_code == 401
    app.dependency_overrides.clear()


def test_protected_endpoint_with_invalid_token_returns_401():
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_user_repo] = lambda: FakeUserRepo()
    app.dependency_overrides[get_ask_question_use_case] = lambda: MagicMock()

    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "anything"},
        headers={"Authorization": "Bearer not-a-token"},
    )

    assert response.status_code == 401
    app.dependency_overrides.clear()


def test_valid_token_passes_guard_and_reaches_use_case():
    user = User(id="u1", email="alice@example.com", hashed_password="x")
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_user_repo] = lambda: FakeUserRepo(user)

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(
        return_value=Answer(
            text="answer",
            sources=[
                Document(id="u1::doc.pdf", content="x", metadata={"filename": "doc.pdf"})
            ],
        )
    )
    app.dependency_overrides[get_ask_question_use_case] = lambda: mock_use_case

    token = create_access_token(user.id)
    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "question"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "answer"
    # the use case received the logged-in user's ID as the owner_id
    assert mock_use_case.execute.call_args.args[1] == "u1"
    app.dependency_overrides.clear()