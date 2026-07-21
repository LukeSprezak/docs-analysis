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
    NONE = "none"  # no reranking — returns top_k straight from vector search (NoOpReranker)
    LLM = "llm"  # listwise reranking by the configured LLM
    COHERE = "cohere"  # cross-encoder via the Cohere Rerank API
    BGE = "bge"  # local cross-encoder (sentence-transformers, offline, no API)


class EvalJudgeProvider(StrEnum):
    NONE = "none"  # generation metrics disabled (eval computes retrieval only, no LLM)
    LLM = "llm"  # faithfulness/answer-relevance scored by the configured LLM


class SearchStrategy(StrEnum):
    VECTOR = "vector"  # vector similarity only (the default)
    HYBRID = "hybrid"  # vectors + keywords (Postgres FTS), combined with RRF
