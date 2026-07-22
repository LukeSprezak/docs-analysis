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
    # Every value comes from the environment (.env or process variables). Fields without a
    # default are REQUIRED — a missing variable raises a validation error at startup
    # (fail fast) instead of quietly using a fallback baked into the code.
    # The exceptions are the provider API keys and LOG_FILE: they stay optional (None =
    # not configured), because only the key of the provider actually in use is set.
    PROJECT_NAME: str
    API_V1_STR: str

    LLM_PROVIDER: LLMProvider
    VECTOR_STORE_PROVIDER: VectorStoreProvider
    RERANKER_PROVIDER: RerankerProvider

    # Backing store for the domain repositories. The application layer only ever sees the
    # ABCs from `domain/repositories.py`; this variable is the single place that decides
    # which adapters get wired in (see `infrastructure/persistence/factory.py`).
    PERSISTENCE_PROVIDER: PersistenceProvider

    # Knowledge graph. `none` disables it entirely — no entity extraction on upload (which
    # costs an LLM call per document) and no graph candidates during retrieval.
    KNOWLEDGE_GRAPH_PROVIDER: KnowledgeGraphProvider

    # Retrieval: how many chunks to pull from vector search (reranking candidates) and how
    # many to finally pass to the LLM after reordering.
    RETRIEVAL_CANDIDATE_COUNT: int
    RETRIEVAL_TOP_K: int

    # Candidate search strategy: vector or hybrid (vectors + Postgres FTS, combined with
    # RRF). Hybrid only works with VECTOR_STORE_PROVIDER=postgres.
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

    # Connection pool (SQLAlchemy async engine). pool_size = permanent connections kept open,
    # max_overflow = how many extra may be opened under peak load.
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int

    # Neo4j. Optional for the same reason the provider API keys are: they only have to be set
    # when Neo4j is the selected provider. The factory checks them then and fails loudly, so a
    # Postgres-only deployment does not have to carry graph credentials.
    NEO4J_URI: str | None = None
    NEO4J_USERNAME: str | None = None
    NEO4J_PASSWORD: str | None = None
    NEO4J_DATABASE: str | None = None  # None = the server's default database ("neo4j")

    LOG_LEVEL: str
    LOG_FORMAT: str  # TEXT or JSON
    LOG_FILE: str | None = None  # log file path; None = log to stdout only

    MAX_UPLOAD_SIZE_MB: int

    # Upload validation: allowlist of extensions (comma-separated, with a dot). A file outside
    # the list is rejected before anything is written. For PDFs we additionally check the
    # `%PDF-` header (magic bytes) — the extension and declared content-type are controlled by
    # the client and are not trusted.
    ALLOWED_UPLOAD_EXTENSIONS: str

    # List pagination (documents/summaries/conversations): default and maximum page size.
    # Without a limit `list_all` pulled every row belonging to the owner — a risk on a large
    # database (memory + query time).
    LIST_DEFAULT_LIMIT: int
    LIST_MAX_LIMIT: int

    # CORS: comma-separated list of origins the browser may send credentialed requests from.
    # NEVER "*" together with allow_credentials (an invalid and dangerous combination).
    CORS_ALLOWED_ORIGINS: str

    # Rate limiting (slowapi, per IP). Guards against DoS and runaway LLM costs. Disabled in
    # tests (autouse fixture) so repeated calls do not run into the limit. Limit format per
    # the `limits` library: "<count>/<window>", e.g. "20/minute".
    RATE_LIMIT_ENABLED: bool
    RATE_LIMIT_DEFAULT: str  # global safety net for the remaining endpoints
    RATE_LIMIT_LLM: str  # qa/ask, chat, chat/stream — expensive LLM calls
    RATE_LIMIT_UPLOAD: str  # upload — parsing + embedding are costly
    RATE_LIMIT_AUTH: str  # login/register — anti brute force

    # Authentication (JWT). The SECRET MUST be set via env — long (>=32 bytes) and random in
    # production. A missing variable stops the application from starting.
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
