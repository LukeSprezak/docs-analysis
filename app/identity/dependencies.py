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
    if credentials is None:
        raise AuthenticationException("Missing authentication token")
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise AuthenticationException("Invalid or expired token")
    user = await user_repo.get_by_id(user_id)
    if user is None:
        raise AuthenticationException("User no longer exists")
    return user
