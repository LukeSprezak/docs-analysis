from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.shared.config import settings
from app.shared.enums import LLMProvider


def _as_secret(api_key: str | None) -> SecretStr | None:
    """Klucz API jako SecretStr (typ oczekiwany przez klientów LangChain). None → None,
    żeby biblioteka mogła sięgnąć po klucz ze zmiennej środowiskowej."""
    return SecretStr(api_key) if api_key else None


class EmbeddingsFactory:
    @staticmethod
    def get_embeddings() -> Embeddings:
        provider = settings.LLM_PROVIDER.lower()

        match provider:
            case LLMProvider.OPENAI:
                return OpenAIEmbeddings(api_key=_as_secret(settings.OPENAI_API_KEY))
            case LLMProvider.GOOGLE:
                return GoogleGenerativeAIEmbeddings(
                    google_api_key=settings.GOOGLE_API_KEY,  # type: ignore[call-arg]
                    model="models/gemini-embedding-001",
                )
            case LLMProvider.OLLAMA:
                return OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model="llama3")
            case LLMProvider.ANTHROPIC:
                if settings.OPENAI_API_KEY:
                    return OpenAIEmbeddings(api_key=_as_secret(settings.OPENAI_API_KEY))
                raise ValueError(
                    "Anthropic provider selected but no Embeddings fallback available. Please provide OPENAI_API_KEY."
                )
            case _:
                raise ValueError(f"Unsupported Embeddings provider: {provider}")
