"""Selects the adapters for the retrieval context: vector store, knowledge graph, extractor.

This module is the *only* place that names a concrete repository class. Everything above it
(`app.retrieval.dependencies`, the pipeline, the routers) sees nothing but the Protocols from
`ports.py`, so swapping a backing store means adding a class here and a member
to the provider enum — no caller changes.

The choice is global and comes from the environment (`VECTOR_STORE_PROVIDER`,
`KNOWLEDGE_GRAPH_PROVIDER`), matching the existing configuration style: an explicit provider,
no silent fallback. An unsupported combination raises at startup rather than degrading into
a half-working system.
"""

from typing import NamedTuple

from app.retrieval.embedder import create_embeddings, embedding_model_id
from app.retrieval.knowledge_graph.entity_extractor import LLMEntityExtractor
from app.retrieval.knowledge_graph.neo4j import Neo4jKnowledgeGraphRepo
from app.retrieval.knowledge_graph.null import NullEntityExtractor, NullKnowledgeGraphRepo
from app.retrieval.ports import EntityExtractor, KnowledgeGraphRepo, VectorStoreRepo
from app.retrieval.vector_store.faiss import FaissVectorStoreRepo
from app.retrieval.vector_store.neo4j import Neo4jVectorStoreRepo
from app.retrieval.vector_store.postgres import PostgresVectorStoreRepo
from app.shared.config import settings
from app.shared.enums import (
    KnowledgeGraphProvider,
    SearchStrategy,
    VectorStoreProvider,
)
from app.shared.llm.llm_factory import LLMFactory


def create_vector_store_repo() -> VectorStoreRepo:
    embeddings = create_embeddings()
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.FAISS:
        # Hybrid retrieval needs a keyword index alongside the vectors; the in-memory store
        # has none, so FAISS stays vector-only regardless of RETRIEVAL_STRATEGY.
        return FaissVectorStoreRepo(embeddings=embeddings)
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.POSTGRES:
        return PostgresVectorStoreRepo(
            embeddings=embeddings,
            embedding_model=embedding_model_id(embeddings),
            enable_hybrid_search=settings.RETRIEVAL_STRATEGY == SearchStrategy.HYBRID,
        )
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.NEO4J:
        credentials = _neo4j_credentials()
        return Neo4jVectorStoreRepo(
            embeddings=embeddings,
            embedding_model=embedding_model_id(embeddings),
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
