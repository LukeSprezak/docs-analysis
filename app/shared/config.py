from pydantic_settings import BaseSettings, SettingsConfigDict

from app.shared.enums import (
    EvalJudgeProvider,
    LLMProvider,
    RerankerProvider,
    SearchStrategy,
    VectorStoreProvider,
)


class Settings(BaseSettings):
    PROJECT_NAME: str
    API_V1_STR: str

    LLM_PROVIDER: LLMProvider
    VECTOR_STORE_PROVIDER: VectorStoreProvider
    RERANKER_PROVIDER: RerankerProvider

    # Retrieval: how many segments to get from vector search (candidates for rerankingu) and how much
    # finally, forward it to LLM after sorting.
    RETRIEVAL_CANDIDATE_COUNT: int
    RETRIEVAL_TOP_K: int

    # Candidate Search Strategy: vector lub hybrid (wektory + Postgres FTS,
    # combined RRF). Hybrid works only for VECTOR_STORE_PROVIDER=postgres.
    RETRIEVAL_STRATEGY: SearchStrategy

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    OLLAMA_BASE_URL: str
    LLM_MODEL: str | None = None  # e.g. gpt-4o, claude-3-opus, etc.

    COHERE_API_KEY: str | None = None
    COHERE_RERANK_MODEL: str  # multilingual (supports Polish)

    # Local cross-encoder (RERANKER_PROVIDER=bge); required extra `local-reranker`.
    BGE_RERANKER_MODEL: str  # multilingual (supports Polish)

    # Eval harness (AI-9). Retrieval metrics always matter (offline, without LLM);
    # generation metrics (faithfulness/answer-relevance) require an LLM judge.
    EVAL_JUDGE_PROVIDER: EvalJudgeProvider
    EVAL_DATASET_PATH: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Connection pool (SQLAlchemy async engine). pool_size = persistent connections kept open,
    # max_overflow = How many additional ones can be opened during peak load
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int

    LOG_LEVEL: str
    LOG_FORMAT: str  # TEXT or JSON
    LOG_FILE: str | None = None  # log file path; None = log only to stdout

    MAX_UPLOAD_SIZE_MB: int

    ALLOWED_UPLOAD_EXTENSIONS: str

    LIST_DEFAULT_LIMIT: int
    LIST_MAX_LIMIT: int

    CORS_ALLOWED_ORIGINS: str

    RATE_LIMIT_ENABLED: bool
    RATE_LIMIT_DEFAULT: str  # global safety net for other endpoints
    RATE_LIMIT_LLM: str  # qa/ask, chat, chat/stream — expensive LLM calls
    RATE_LIMIT_UPLOAD: str  # upload — parsing and embedding are computationally intensive
    RATE_LIMIT_AUTH: str  # login/register — anty-brute-force

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def database_url(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sync_database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()  # type: ignore[call-arg]
