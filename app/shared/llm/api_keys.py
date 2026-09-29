"""Shared handling of API keys for the LangChain factories."""

from pydantic import SecretStr


def as_secret(api_key: str | None) -> SecretStr | None:
    """The API key as SecretStr (the type LangChain clients expect). None → None, so the
    library can fall back to reading the key from an environment variable."""
    return SecretStr(api_key) if api_key else None
