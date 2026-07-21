from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.shared.config import settings
from app.shared.enums import LLMProvider


def _as_secret(api_key: str | None) -> SecretStr | None:
    """The API key as SecretStr (the type LangChain clients expect). None → None, so the
    library can fall back to reading the key from an environment variable."""
    return SecretStr(api_key) if api_key else None


class LLMFactory:
    @staticmethod
    def get_llm() -> BaseChatModel:
        provider = settings.LLM_PROVIDER.lower()

        match provider:
            case LLMProvider.OPENAI:
                return ChatOpenAI(
                    api_key=_as_secret(settings.OPENAI_API_KEY),
                    model=settings.LLM_MODEL or "gpt-4o-mini",
                )
            case LLMProvider.ANTHROPIC:
                # langchain_anthropic types api_key as a required SecretStr and timeout/stop as
                # required — at runtime they default to None, and a missing key is read from an
                # environment variable. timeout/stop=None == default; the ignore covers only the
                # incorrect api_key type (Optional at runtime, non-Optional in the stub).
                return ChatAnthropic(
                    api_key=_as_secret(settings.ANTHROPIC_API_KEY),  # type: ignore[arg-type]
                    model_name=settings.LLM_MODEL or "claude-3-5-sonnet-20240620",
                    timeout=None,
                    stop=None,
                )
            case LLMProvider.GOOGLE:
                return ChatGoogleGenerativeAI(
                    google_api_key=settings.GOOGLE_API_KEY,
                    model=settings.LLM_MODEL or "gemini-1.5-flash",
                )
            case LLMProvider.OLLAMA:
                return ChatOllama(
                    base_url=settings.OLLAMA_BASE_URL, model=settings.LLM_MODEL or "llama3"
                )
            case _:
                raise ValueError(f"Unsupported LLM provider: {provider}")
