from enum import StrEnum


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class VectorStoreProvider(StrEnum):
    POSTGRES = "postgres"
    FAISS = "faiss"


class RerankerProvider(StrEnum):
    NONE = "none"  # bez rerankingu — zwraca top_k z vector search (NoOpReranker)
    LLM = "llm"  # reranking listwise przez skonfigurowany LLM
    COHERE = "cohere"  # cross-encoder przez Cohere Rerank API
    BGE = "bge"  # lokalny cross-encoder (sentence-transformers, offline, bez API)


class EvalJudgeProvider(StrEnum):
    NONE = "none"  # metryki generacji wyłączone (eval liczy tylko retrieval, bez LLM)
    LLM = "llm"  # faithfulness/answer-relevance oceniane przez skonfigurowany LLM


class SearchStrategy(StrEnum):
    VECTOR = "vector"  # tylko podobieństwo wektorowe (domyślnie)
    HYBRID = "hybrid"  # wektory + słowa kluczowe (Postgres FTS), łączone RRF
