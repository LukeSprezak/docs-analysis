"""Prymitywy bezpieczeństwa kontekstu Identity: hashowanie haseł (bcrypt) i tokeny JWT.

Czyste, bezstanowe funkcje używane przez wszystkie warstwy tego kontekstu (use case'y
hashują/weryfikują hasło, router i guard tworzą/dekodują token). Dlatego leżą w korzeniu
kontekstu, a nie w `infrastructure/` — nie są wymiennym adapterem ani modelem domenowym,
tylko współdzielonym narzędziem kryptograficznym. Trzymane razem, żeby logika kryptografii
była w jednym miejscu i testowalna bez sieci i bazy.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.shared.config import settings


def hash_password(plain_password: str) -> str:
    """Zwraca hash bcrypt hasła (z solą) jako tekst do zapisania w bazie."""
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed_bytes.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Sprawdza, czy hasło pasuje do zapisanego hasha. Fail-closed na uszkodzonym hashu."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str) -> str:
    """Buduje podpisany token dostępu (HS256) z `sub` = id użytkownika i czasem wygaśnięcia."""
    issued_at = datetime.now(UTC)
    expires_at = issued_at + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "iat": issued_at, "exp": expires_at}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Weryfikuje podpis i wygaśnięcie; zwraca id użytkownika (`sub`) albo None, gdy
    token jest nieprawidłowy/wygasły. Nie rzuca — wołający decyduje, jak zareagować."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    subject = payload.get("sub")
    if not isinstance(subject, str):
        return None
    return subject
