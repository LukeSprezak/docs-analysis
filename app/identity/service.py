from uuid import uuid4

from app.identity.models import User
from app.identity.repo import UserRepo
from app.identity.security import hash_password, verify_password
from app.shared.exceptions import AuthenticationException, ValidationException


async def register_user(user_repo: UserRepo, email: str, password: str) -> User:
    normalized_email = email.strip().lower()
    if await user_repo.get_by_email(normalized_email) is not None:
        raise ValidationException("Email is already registered")

    user = User(
        id=str(uuid4()),
        email=normalized_email,
        hashed_password=hash_password(password),
    )
    await user_repo.save(user)
    return user


async def authenticate_user(user_repo: UserRepo, email: str, password: str) -> User:
    normalized_email = email.strip().lower()
    user = await user_repo.get_by_email(normalized_email)

    if user is None or not verify_password(password, user.hashed_password):
        raise AuthenticationException("Invalid email or password")
    return user
