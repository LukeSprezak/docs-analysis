"""Dependency injection for the retrieval context.

Every provider is typed against the port, never a concrete adapter; the only code that names
adapters is `factory.py` (plus `create_reranker` in `reranker.py`). The repositories are process-wide singletons built
eagerly by `init()` in the application lifespan, and lazily (under a lock) for callers that
run without one, such as the evaluation harness.
"""

import threading
from typing import Annotated

from fastapi import Depends

from app.retrieval import factory
from app.retrieval.indexing import IndexDocumentUseCase, RemoveFromIndexUseCase
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.ports import (
    EntityExtractor,
    KnowledgeGraphRepo,
    RerankerService,
    VectorStoreRepo,
)
from app.retrieval.rag_service import LangChainRAGService
from app.retrieval.reranker import create_reranker
from app.shared.config import settings
from app.shared.llm.llm_factory import LLMFactory

_singleton_lock = threading.Lock()

_vector_repo: VectorStoreRepo | None = None
_graph_repo: KnowledgeGraphRepo | None = None
_entity_extractor: EntityExtractor | None = None


def get_vector_repo() -> VectorStoreRepo:
    global _vector_repo
    with _singleton_lock:
        if _vector_repo is None:
            _vector_repo = factory.create_vector_store_repo()
        return _vector_repo


def get_graph_repo() -> KnowledgeGraphRepo:
    global _graph_repo
    with _singleton_lock:
        if _graph_repo is None:
            _graph_repo = factory.create_knowledge_graph_repo()
        return _graph_repo


def get_entity_extractor() -> EntityExtractor:
    global _entity_extractor
    with _singleton_lock:
        if _entity_extractor is None:
            _entity_extractor = factory.create_entity_extractor()
        return _entity_extractor


def init() -> None:
    """Builds the singletons up front, so an unreachable Neo4j fails startup, not a query."""
    get_vector_repo()
    get_graph_repo()
    get_entity_extractor()


async def shutdown() -> None:
    """Closes the drivers the vector store and the graph may own, and clears the singletons so
    a later startup in the same process (tests, an ASGI reload) builds them fresh."""
    global _vector_repo, _graph_repo, _entity_extractor
    if _vector_repo is not None:
        await _vector_repo.close()
    if _graph_repo is not None:
        await _graph_repo.close()
    _vector_repo = _graph_repo = _entity_extractor = None


def get_rag_service() -> LangChainRAGService:
    return LangChainRAGService(llm=LLMFactory.get_llm())


def get_reranker_service() -> RerankerService:
    return create_reranker()


def get_retrieval_pipeline(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    reranker: Annotated[RerankerService, Depends(get_reranker_service)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> RetrievalPipeline:
    return RetrievalPipeline(
        vector_repo,
        reranker,
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
        graph_repo=graph_repo,
    )


def get_index_document_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
    entity_extractor: Annotated[EntityExtractor, Depends(get_entity_extractor)],
) -> IndexDocumentUseCase:
    return IndexDocumentUseCase(vector_repo, graph_repo, entity_extractor)


def get_remove_from_index_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> RemoveFromIndexUseCase:
    return RemoveFromIndexUseCase(vector_repo, graph_repo)
