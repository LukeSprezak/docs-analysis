from uuid import uuid4

from app.identity.domain.models import User
from app.identity.domain.repositories import UserRepo
from app.identity.security import hash_password
from app.shared.exceptions import ValidationException


class RegisterUserUseCase:
    def __init__(self, user_repo: UserRepo):
        self.user_repo = user_repo

    async def execute(self, email: str, password: str) -> User:
        normalized_email = email.strip().lower()
        if await self.user_repo.get_by_email(normalized_email) is not None:
            raise ValidationException("Email is already registered")

        user = User(
            id=str(uuid4()),
            email=normalized_email,
            hashed_password=hash_password(password),
        )
        await self.user_repo.save(user)
        return user