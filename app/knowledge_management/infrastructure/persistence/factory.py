"""Selects the persistence adapters for the knowledge_management context.

This module is the *only* place that names a concrete repository class. Everything above it
(`app.shared.dependencies`, the use cases, the routers) sees nothing but the ABCs from
`domain/repositories.py`, so swapping a backing store means adding a class here and a member
to the provider enum — no caller changes.

The choice is global and comes from the environment (`PERSISTENCE_PROVIDER`,
`VECTOR_STORE_PROVIDER`), matching the existing configuration style: an explicit provider,
no silent fallback. An unsupported combination raises at startup rather than degrading into
a half-working system.
"""

from typing import NamedTuple

from app.knowledge_management.domain.null_entity_extractor import NullEntityExtractor
from app.knowledge_management.domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from app.knowledge_management.domain.repositories import (
    ConversationRepo,
    DocumentRepo,
    EntityExtractor,
    KnowledgeGraphRepo,
    SummaryRepo,
    VectorStoreRepo,
)
from app.knowledge_management.infrastructure.llm.embeddings_factory import EmbeddingsFactory
from app.knowledge_management.infrastructure.llm.entity_extractor import LLMEntityExtractor
from app.knowledge_management.infrastructure.llm.llm_factory import LLMFactory
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import (
    FaissVectorStoreRepo,
)
from app.knowledge_management.infrastructure.persistence.neo4j_knowledge_graph_repo import (
    Neo4jKnowledgeGraphRepo,
)
from app.knowledge_management.infrastructure.persistence.neo4j_vectorstore_repo import (
    Neo4jVectorStoreRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_conversation_repo import (
    PostgresConversationRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_document_repo import (
    PostgresDocumentRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_summary_repo import (
    PostgresSummaryRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_vectorstore_repo import (
    PostgresVectorStoreRepo,
)
from app.shared.config import settings
from app.shared.enums import (
    KnowledgeGraphProvider,
    PersistenceProvider,
    SearchStrategy,
    VectorStoreProvider,
)


def create_document_repo() -> DocumentRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresDocumentRepo()
    raise _unsupported("DocumentRepo", settings.PERSISTENCE_PROVIDER)


def create_summary_repo() -> SummaryRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresSummaryRepo()
    raise _unsupported("SummaryRepo", settings.PERSISTENCE_PROVIDER)


def create_conversation_repo() -> ConversationRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresConversationRepo()
    raise _unsupported("ConversationRepo", settings.PERSISTENCE_PROVIDER)


def create_vector_store_repo() -> VectorStoreRepo:
    embeddings = EmbeddingsFactory.get_embeddings()
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.FAISS:
        # Hybrid retrieval needs a keyword index alongside the vectors; the in-memory store
        # has none, so FAISS stays vector-only regardless of RETRIEVAL_STRATEGY.
        return FaissVectorStoreRepo(embeddings=embeddings)
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.POSTGRES:
        return PostgresVectorStoreRepo(
            embeddings=embeddings,
            enable_hybrid_search=settings.RETRIEVAL_STRATEGY == SearchStrategy.HYBRID,
        )
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.NEO4J:
        credentials = _neo4j_credentials()
        return Neo4jVectorStoreRepo(
            embeddings=embeddings,
            enable_hybrid_search=settings.RETRIEVAL_STRATEGY == SearchStrategy.HYBRID,
            url=credentials.url,
            username=credentials.username,
            password=credentials.password,
            database=credentials.database,
        )
    raise _unsupported("VectorStoreRepo", settings.VECTOR_STORE_PROVIDER)


def create_knowledge_graph_repo() -> KnowledgeGraphRepo:
    if settings.KNOWLEDGE_GRAPH_PROVIDER == KnowledgeGraphProvider.NONE:
        return NullKnowledgeGraphRepo()
    if settings.KNOWLEDGE_GRAPH_PROVIDER == KnowledgeGraphProvider.NEO4J:
        credentials = _neo4j_credentials()
        return Neo4jKnowledgeGraphRepo(
            url=credentials.url,
            username=credentials.username,
            password=credentials.password,
            database=credentials.database,
        )
    raise _unsupported("KnowledgeGraphRepo", settings.KNOWLEDGE_GRAPH_PROVIDER)


def create_entity_extractor() -> EntityExtractor:
    """Pairs with the graph repository: no graph means no extraction, hence no LLM cost."""
    if settings.KNOWLEDGE_GRAPH_PROVIDER == KnowledgeGraphProvider.NONE:
        return NullEntityExtractor()
    return LLMEntityExtractor(llm=LLMFactory.get_llm())


class Neo4jCredentials(NamedTuple):
    url: str
    username: str
    password: str
    database: str | None


def _neo4j_credentials() -> Neo4jCredentials:
    """Neo4j connection settings, checked here rather than at import time.

    The variables are optional in `Settings` so a Postgres-only deployment needs no graph
    credentials; the moment Neo4j is actually selected, a missing one has to stop startup
    instead of surfacing later as an authentication error on the first query. Narrowing them
    to `str` here is also what keeps the adapter's signature free of `| None`.
    """
    url, username, password = settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD
    if not url or not username or not password:
        missing = [
            name
            for name, value in (
                ("NEO4J_URI", url),
                ("NEO4J_USERNAME", username),
                ("NEO4J_PASSWORD", password),
            )
            if not value
        ]
        raise ValueError(f"Selecting neo4j requires {', '.join(missing)} to be set")
    return Neo4jCredentials(url, username, password, settings.NEO4J_DATABASE)


def _unsupported(port_name: str, provider: str) -> NotImplementedError:
    return NotImplementedError(f"No {port_name} adapter for provider '{provider}'")
