from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.shared.config import settings
from app.shared.enums import LLMProvider
from app.shared.llm.api_keys import as_secret

ProviderEmbeddings = OpenAIEmbeddings | GoogleGenerativeAIEmbeddings | OllamaEmbeddings


def embedding_model_id(embeddings: ProviderEmbeddings) -> str:
    """What an index records as the model that built it, e.g. `OpenAIEmbeddings:text-embedding-ada-002`.

    Vectors from two models are not comparable, so an index built with one and queried with
    another returns nonsense instead of an error — the id is what lets the stores refuse."""
    return f"{type(embeddings).__name__}:{embeddings.model}"


def create_embeddings() -> ProviderEmbeddings:
    provider = settings.LLM_PROVIDER.lower()

    match provider:
        case LLMProvider.OPENAI:
            return OpenAIEmbeddings(api_key=as_secret(settings.OPENAI_API_KEY))
        case LLMProvider.GOOGLE:
            return GoogleGenerativeAIEmbeddings(
                google_api_key=settings.GOOGLE_API_KEY,  # type: ignore[call-arg]
                model="models/gemini-embedding-001",
            )
        case LLMProvider.OLLAMA:
            return OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model="llama3")
        case LLMProvider.ANTHROPIC:
            if settings.OPENAI_API_KEY:
                return OpenAIEmbeddings(api_key=as_secret(settings.OPENAI_API_KEY))
            raise ValueError(
                "Anthropic provider selected but no Embeddings fallback available. Please provide OPENAI_API_KEY."
            )
        case _:
            raise ValueError(f"Unsupported Embeddings provider: {provider}")
