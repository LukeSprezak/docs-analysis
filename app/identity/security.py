"""Security primitives of the Identity context: password hashing (bcrypt) and JWT tokens.

Pure, stateless functions used by every layer of this context (use cases hash/verify the
password, the router and the guard create/decode the token). They live at the context root
rather than in `infrastructure/` because they are neither a swappable adapter nor a domain
model, just a shared cryptographic tool. Kept together so the crypto logic sits in one place
and stays testable without network or database.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.shared.config import settings


def hash_password(plain_password: str) -> str:
    """Returns the bcrypt hash of the password (salted), as text to store in the database."""
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed_bytes.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks the password against the stored hash. Fails closed on a corrupted hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str) -> str:
    """Builds a signed access token (HS256) with `sub` = user id and an expiry time."""
    issued_at = datetime.now(UTC)
    expires_at = issued_at + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "iat": issued_at, "exp": expires_at}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Verifies the signature and expiry; returns the user id (`sub`), or None when the token
    is invalid/expired. Never raises — the caller decides how to react."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    subject = payload.get("sub")
    if not isinstance(subject, str):
        return None
    return subject
