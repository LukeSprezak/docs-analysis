from enum import StrEnum


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class VectorStoreProvider(StrEnum):
    POSTGRES = "postgres"
    FAISS = "faiss"
    NEO4J = "neo4j"


class KnowledgeGraphProvider(StrEnum):
    """Backing store for the knowledge graph, independent of where the vectors live.

    `none` is a real choice, not just "unconfigured": it wires in the null repository and the
    null extractor, so uploads skip the (expensive) entity-extraction LLM call and retrieval
    stays vector-only — without any branch in the use cases.
    """

    NONE = "none"
    NEO4J = "neo4j"


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
