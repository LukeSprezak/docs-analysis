import pytest

from app.identity.application.authenticate_user import AuthenticateUserUseCase
from app.identity.application.register_user import RegisterUserUseCase
from app.identity.domain.models import User
from app.identity.security import hash_password
from app.shared.exceptions import AuthenticationException, ValidationException


class FakeUserRepo:
    def __init__(self):
        self.users_by_email: dict[str, User] = {}

    async def get_by_email(self, email: str) -> User | None:
        return self.users_by_email.get(email)

    async def get_by_id(self, user_id: str) -> User | None:
        for user in self.users_by_email.values():
            if user.id == user_id:
                return user
        return None

    async def save(self, user: User) -> None:
        self.users_by_email[user.email] = user


async def test_register_creates_user_with_hashed_password():
    repo = FakeUserRepo()
    use_case = RegisterUserUseCase(repo)

    user = await use_case.execute("Alice@Example.com", "secret-password")

    #  email converted to lowercase
    assert user.email == "alice@example.com"
    # the password is not stored in plain text
    assert user.hashed_password != "secret-password"
    assert await repo.get_by_email("alice@example.com") is user


async def test_register_rejects_duplicate_email():
    repo = FakeUserRepo()
    use_case = RegisterUserUseCase(repo)
    await use_case.execute("alice@example.com", "password")

    with pytest.raises(ValidationException):
        await use_case.execute("alice@example.com", "other")


async def test_authenticate_returns_user_for_valid_credentials():
    repo = FakeUserRepo()
    await repo.save(
        User(id="u1", email="alice@example.com", hashed_password=hash_password("password"))
    )
    use_case = AuthenticateUserUseCase(repo)

    user = await use_case.execute("alice@example.com", "password")
    assert user.id == "u1"


async def test_authenticate_rejects_wrong_password():
    repo = FakeUserRepo()
    await repo.save(
        User(id="u1", email="alice@example.com", hashed_password=hash_password("password"))
    )
    use_case = AuthenticateUserUseCase(repo)

    with pytest.raises(AuthenticationException):
        await use_case.execute("alice@example.com", "wrong-password")


async def test_authenticate_rejects_unknown_user():
    use_case = AuthenticateUserUseCase(FakeUserRepo())
    with pytest.raises(AuthenticationException):
        await use_case.execute("nieznany@example.com", "haslo")