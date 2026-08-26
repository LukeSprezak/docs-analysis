from pydantic_settings import BaseSettings, SettingsConfigDict

from app.shared.enums import (
    EvalJudgeProvider,
    KnowledgeGraphProvider,
    LLMProvider,
    PersistenceProvider,
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

    PERSISTENCE_PROVIDER: PersistenceProvider

    KNOWLEDGE_GRAPH_PROVIDER: KnowledgeGraphProvider

    RETRIEVAL_CANDIDATE_COUNT: int
    RETRIEVAL_TOP_K: int

    RETRIEVAL_STRATEGY: SearchStrategy

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    OLLAMA_BASE_URL: str
    LLM_MODEL: str | None = None  # e.g. gpt-4o, claude-3-opus, etc.

    COHERE_API_KEY: str | None = None
    COHERE_RERANK_MODEL: str  # multilingual (supports Polish)

    # Local cross-encoder (RERANKER_PROVIDER=bge); requires the `local-reranker` extra.
    BGE_RERANKER_MODEL: str  # multilingual (supports Polish)

    # Eval harness (AI-9). Retrieval metrics are always computed (offline, no LLM);
    # generation metrics (faithfulness/answer-relevance) require an LLM judge.
    EVAL_JUDGE_PROVIDER: EvalJudgeProvider
    EVAL_DATASET_PATH: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int

    NEO4J_URI: str | None = None
    NEO4J_USERNAME: str | None = None
    NEO4J_PASSWORD: str | None = None
    NEO4J_DATABASE: str | None = None  # None = the server's default database ("neo4j")

    LOG_LEVEL: str
    LOG_FORMAT: str  # TEXT or JSON
    LOG_FILE: str | None = None  # log file path; None = log to stdout only

    MAX_UPLOAD_SIZE_MB: int

    ALLOWED_UPLOAD_EXTENSIONS: str

    LIST_DEFAULT_LIMIT: int
    LIST_MAX_LIMIT: int

    CORS_ALLOWED_ORIGINS: str

    # Rate limiting (slowapi, per IP). Guards against DoS and runaway LLM costs. Disabled in
    # tests (autouse fixture) so repeated calls do not run into the limit. Limit format per
    # the `limits` library: "<count>/<window>", e.g. "20/minute".
    RATE_LIMIT_ENABLED: bool
    RATE_LIMIT_DEFAULT: str  # global safety net for the remaining endpoints
    RATE_LIMIT_LLM: str  # qa/ask, chat, chat/stream — expensive LLM calls
    RATE_LIMIT_UPLOAD: str  # upload — parsing + embedding are costly
    RATE_LIMIT_AUTH: str  # login/register — anti brute force

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """CORS origins as a list (parsed from env by comma, empty entries skipped)."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def database_url(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sync_database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


# Required fields are filled from env/.env at runtime — mypy cannot see that and reports
# missing constructor arguments, hence the deliberate silencing.
settings = Settings()  # type: ignore[call-arg]
