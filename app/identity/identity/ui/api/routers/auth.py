from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.identity.application.authenticate_user import AuthenticateUserUseCase
from app.identity.application.register_user import RegisterUserUseCase
from app.identity.dependencies import (
    get_authenticate_user_use_case,
    get_register_user_use_case,
)
from app.identity.security import create_access_token
from app.shared.config import settings
from app.shared.rate_limit import limiter

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterCommand(BaseModel):
    email: str
    password: str


class LoginCommand(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 — typ tokenu OAuth2, nie sekret
    user_id: str
    email: str


@router.post("/register", response_model=TokenResponse, status_code=201)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(
    request: Request,
    command: RegisterCommand,
    use_case: Annotated[RegisterUserUseCase, Depends(get_register_user_use_case)],
) -> TokenResponse:
    user = await use_case.execute(command.email, command.password)
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        email=user.email,
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(
    request: Request,
    command: LoginCommand,
    use_case: Annotated[AuthenticateUserUseCase, Depends(get_authenticate_user_use_case)],
) -> TokenResponse:
    user = await use_case.execute(command.email, command.password)
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        email=user.email,
    )
