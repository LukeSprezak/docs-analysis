from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.shared.config import settings
from app.shared.enums import LLMProvider

from .api_keys import as_secret


class LLMFactory:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_llm() -> BaseChatModel:
        """The chat model, built once per process.

        The clients are stateless, so a new one per request bought nothing and cost a fresh
        `httpx.Client` each time — every call paying for a new TLS handshake instead of
        reusing the provider connection. Cached like `_load_bge_scorer` and like the
        repository singletons; `settings` is read once at import, so nothing here can go
        stale.
        """
        provider = settings.LLM_PROVIDER.lower()

        match provider:
            case LLMProvider.OPENAI:
                return ChatOpenAI(
                    api_key=as_secret(settings.OPENAI_API_KEY),
                    model=settings.LLM_MODEL or "gpt-4o-mini",
                )
            case LLMProvider.ANTHROPIC:
                # langchain_anthropic types api_key as a required SecretStr and timeout/stop as
                # required — at runtime they default to None, and a missing key is read from an
                # environment variable. timeout/stop=None == default; the ignore covers only the
                # incorrect api_key type (Optional at runtime, non-Optional in the stub).
                return ChatAnthropic(
                    api_key=as_secret(settings.ANTHROPIC_API_KEY),  # type: ignore[arg-type]
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
