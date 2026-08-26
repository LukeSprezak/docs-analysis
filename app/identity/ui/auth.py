from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, EmailStr, Field, field_validator

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


MAX_PASSWORD_BYTES = 72

MIN_PASSWORD_LENGTH = 8


class RegisterCommand(BaseModel):
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)

    @field_validator("password")
    @classmethod
    def _fits_bcrypt(cls, value: str) -> str:
        # Counted in bytes, not characters: 72 Polish characters are ~144 bytes in UTF-8.
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes long")
        return value


class LoginCommand(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 — type token OAuth2, not secret
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
        command: LoginCommand,
    use_case: Annotated[AuthenticateUserUseCase, Depends(get_authenticate_user_use_case)],
) -> TokenResponse:
    user = await use_case.execute(command.email, command.password)
    return TokenResponse(
        access_token=create_access_token(user.id),
        user_id=user.id,
        email=user.email,
    )
