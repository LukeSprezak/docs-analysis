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

from app.knowledge_management.domain.repositories import (
    ConversationRepo,
    DocumentRepo,
    SummaryRepo,
    VectorStoreRepo,
)
from app.knowledge_management.infrastructure.llm.embeddings_factory import EmbeddingsFactory
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import (
    FaissVectorStoreRepo,
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
from app.shared.enums import PersistenceProvider, SearchStrategy, VectorStoreProvider


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
    raise _unsupported("VectorStoreRepo", settings.VECTOR_STORE_PROVIDER)


def _unsupported(port_name: str, provider: str) -> NotImplementedError:
    return NotImplementedError(f"No {port_name} adapter for provider '{provider}'")
