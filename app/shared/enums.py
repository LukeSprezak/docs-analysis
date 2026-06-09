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
    NONE = "none"  # without re-ranking — returns top_k z vector search (NoOpReranker)
    LLM = "llm"  # reranking listwise using a pre-trained LLM
    COHERE = "cohere"  # cross-encoder via the Cohere Rerank API
    BGE = "bge"  # local cross-encoder (sentence-transformers, offline, not API)


class EvalJudgeProvider(StrEnum):
    NONE = "none"  # generation metrics disabled (eval counts only retrieval, without LLM)
    LLM = "llm"  # faithfulness/answer-relevance evaluated by a pre-trained LLM


class SearchStrategy(StrEnum):
    VECTOR = "vector"  # vector similarity only (default)
    HYBRID = "hybrid"  # vectors + keywords (Postgres FTS), combined RRF
