from pydantic_settings import BaseSettings, SettingsConfigDict

from app.shared.enums import (
    EvalJudgeProvider,
    LLMProvider,
    RerankerProvider,
    SearchStrategy,
    VectorStoreProvider,
)


class Settings(BaseSettings):
    # Wszystkie wartości pochodzą ze środowiska (.env lub zmienne procesu). Pola bez
    # wartości domyślnej są WYMAGANE — brak którejkolwiek zmiennej wywoła błąd walidacji
    # przy starcie (fail-fast), zamiast cicho użyć zaszytego w kodzie fallbacku.
    # Wyjątkiem są klucze API providerów oraz LOG_FILE: pozostają opcjonalne (None =
    # nieskonfigurowane), bo konfiguruje się tylko klucz faktycznie używanego providera.
    PROJECT_NAME: str
    API_V1_STR: str

    LLM_PROVIDER: LLMProvider
    VECTOR_STORE_PROVIDER: VectorStoreProvider
    RERANKER_PROVIDER: RerankerProvider

    # Retrieval: ile fragmentów pobrać z vector search (kandydaci do rerankingu) i ile
    # finalnie przekazać do LLM po przesortowaniu.
    RETRIEVAL_CANDIDATE_COUNT: int
    RETRIEVAL_TOP_K: int

    # Strategia wyszukiwania kandydatów: vector lub hybrid (wektory + Postgres FTS,
    # łączone RRF). Hybrid działa tylko dla VECTOR_STORE_PROVIDER=postgres.
    RETRIEVAL_STRATEGY: SearchStrategy

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    OLLAMA_BASE_URL: str
    LLM_MODEL: str | None = None  # e.g. gpt-4o, claude-3-opus, etc.

    COHERE_API_KEY: str | None = None
    COHERE_RERANK_MODEL: str  # multilingual (wspiera polski)

    # Lokalny cross-encoder (RERANKER_PROVIDER=bge); wymaga extra `local-reranker`.
    BGE_RERANKER_MODEL: str  # multilingual (wspiera polski)

    # Eval harness (AI-9). Metryki retrievalu liczą się zawsze (offline, bez LLM);
    # metryki generacji (faithfulness/answer-relevance) wymagają sędziego LLM.
    EVAL_JUDGE_PROVIDER: EvalJudgeProvider
    EVAL_DATASET_PATH: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Pula połączeń (SQLAlchemy async engine). pool_size = stałe połączenia trzymane otwarte,
    # max_overflow = ile dodatkowych można otworzyć pod szczytem obciążenia.
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int

    LOG_LEVEL: str
    LOG_FORMAT: str  # TEXT or JSON
    LOG_FILE: str | None = None  # ścieżka pliku logów; None = logowanie tylko na stdout

    MAX_UPLOAD_SIZE_MB: int

    # Walidacja uploadu: allowlista rozszerzeń (po przecinku, z kropką). Plik spoza listy
    # jest odrzucany zanim cokolwiek zapiszemy. Dla PDF dodatkowo sprawdzamy nagłówek
    # `%PDF-` (magic bytes) — samo rozszerzenie i deklarowany content-type są kontrolowane
    # przez klienta i nie są zaufane.
    ALLOWED_UPLOAD_EXTENSIONS: str

    # Paginacja list (documents/summaries/conversations): domyślny i maksymalny rozmiar
    # strony. Bez limitu `list_all` ciągnęło wszystkie wiersze właściciela — ryzyko przy
    # dużej bazie (pamięć + czas zapytania).
    LIST_DEFAULT_LIMIT: int
    LIST_MAX_LIMIT: int

    # CORS: lista originów (po przecinku), którym przeglądarka może wysyłać żądania z
    # poświadczeniami. NIGDY "*" razem z allow_credentials (niepoprawna i niebezpieczna
    # kombinacja).
    CORS_ALLOWED_ORIGINS: str

    # Rate limiting (slowapi, per-IP). Chroni przed DoS i niekontrolowanymi kosztami LLM.
    # Wyłączane w testach (autouse fixture), żeby powtarzane wywołania nie wpadały w limit.
    # Format limitu wg biblioteki `limits`: "<liczba>/<okno>", np. "20/minute".
    RATE_LIMIT_ENABLED: bool
    RATE_LIMIT_DEFAULT: str  # globalny bezpiecznik dla pozostałych endpointów
    RATE_LIMIT_LLM: str  # qa/ask, chat, chat/stream — drogie wywołania LLM
    RATE_LIMIT_UPLOAD: str  # upload — parsowanie + embedding są kosztowne
    RATE_LIMIT_AUTH: str  # login/register — anty-brute-force

    # Autentykacja (JWT). SEKRET MUSI być ustawiony przez env — w produkcji długi (>=32
    # bajty), losowy. Brak zmiennej zatrzyma start aplikacji.
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Originy CORS jako lista (parsowanie env po przecinku, puste pomijane)."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def database_url(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sync_database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


# Pola wymagane są wypełniane z env/.env w czasie wykonania — mypy tego nie widzi i
# zgłasza brakujące argumenty konstruktora, stąd celowe wyciszenie.
settings = Settings()  # type: ignore[call-arg]
