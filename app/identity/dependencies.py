"""Dependency injection for the Identity context.

`get_current_user` lives here too — the guard exposed as this context's **public
interface**. Other contexts (e.g. knowledge_management) import that guard, receive the
logged-in `User` and read only its `id` as `owner_id`. That makes identity the supplier and
the other contexts its customers, without sharing domain models in the other direction.
"""

from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.identity.application.authenticate_user import AuthenticateUserUseCase
from app.identity.application.register_user import RegisterUserUseCase
from app.identity.domain.models import User
from app.identity.infrastructure.postgres_user_repo import PostgresUserRepo
from app.identity.security import decode_access_token
from app.shared.exceptions import AuthenticationException

_user_repo: PostgresUserRepo | None = None

# Bearer token from the Authorization header. auto_error=False → we raise a consistent
# AuthenticationException (401) instead of FastAPI's default HTTPException.
_bearer_scheme = HTTPBearer(auto_error=False)


def get_user_repo() -> PostgresUserRepo:
    global _user_repo
    if _user_repo is None:
        _user_repo = PostgresUserRepo()
    return _user_repo


def get_register_user_use_case(
    user_repo: Annotated[PostgresUserRepo, Depends(get_user_repo)],
) -> RegisterUserUseCase:
    return RegisterUserUseCase(user_repo)


def get_authenticate_user_use_case(
    user_repo: Annotated[PostgresUserRepo, Depends(get_user_repo)],
) -> AuthenticateUserUseCase:
    return AuthenticateUserUseCase(user_repo)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    user_repo: Annotated[PostgresUserRepo, Depends(get_user_repo)],
) -> User:
    """Extracts and verifies the Bearer token, then loads the user. Raises 401 when the
    token is missing/invalid or the user no longer exists. Every protected endpoint injects
    this dependency to receive the logged-in `User`."""
    if credentials is None:
        raise AuthenticationException("Missing authentication token")
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise AuthenticationException("Invalid or expired token")
    user = await user_repo.get_by_id(user_id)
    if user is None:
        raise AuthenticationException("User no longer exists")
    return user
