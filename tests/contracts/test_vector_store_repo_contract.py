"""Contract suite for the `VectorStoreRepo` port.

Every adapter runs the *same* assertions. This is what makes the backing store genuinely
swappable rather than swappable on paper: a new adapter (Neo4j, Qdrant, …) is only finished
once it is added to `ADAPTERS` below and passes unchanged.

The behaviours pinned here are the ones the use cases and the security model depend on:

* retrieval is isolated per `owner_id` — a query never reaches another user's chunks;
* documents are chunked before embedding, and the parent document id survives the round trip;
* a re-upload replaces the previous chunks instead of orphaning them;
* deleting one document leaves the others intact, and deleting nothing is a no-op.

Adapters backed by a real server are marked `integration` and excluded from the default run
(see `addopts` in pyproject.toml); run them with `pytest -m integration`.
"""

import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.knowledge_management.domain.models import Document
from app.knowledge_management.domain.repositories import VectorStoreRepo
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import (
    FaissVectorStoreRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_vectorstore_repo import (
    PostgresVectorStoreRepo,
)
from app.knowledge_management.infrastructure.text.text_chunker import TextChunker
from app.shared.database import dispose_engine

# A repo builder takes the chunker the test wants (chunk size decides how many fragments a
# document produces) and returns a ready adapter.
RepoFactory = Callable[[TextChunker], VectorStoreRepo]

# Small, deterministic vectors — no network, stable ordering between runs.
EMBEDDING_SIZE = 16


def _embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


@pytest.fixture
def faiss_factory() -> RepoFactory:
    return lambda chunker: FaissVectorStoreRepo(embeddings=_embeddings(), chunker=chunker)


@pytest.fixture
async def postgres_factory() -> AsyncIterator[RepoFactory]:
    """Postgres adapter against a live database, isolated in a throwaway collection.

    Each test gets its own `collection_name`, so repeated runs never see each other's
    vectors. Teardown drops the collections and disposes the shared engine — the pool is a
    module-level singleton bound to the loop that created it, and pytest-asyncio gives each
    test a fresh loop.
    """
    created: list[PostgresVectorStoreRepo] = []

    def build(chunker: TextChunker) -> VectorStoreRepo:
        repo = PostgresVectorStoreRepo(
            embeddings=_embeddings(),
            collection_name=f"contract_test_{uuid.uuid4().hex}",
            chunker=chunker,
        )
        created.append(repo)
        return repo

    yield build

    for repo in created:
        await repo.vector_store.adelete_collection()
    await dispose_engine()


ADAPTERS = [
    pytest.param("faiss_factory", id="faiss"),
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def make_repo(request: pytest.FixtureRequest) -> RepoFactory:
    """The adapter under test, as a builder parametrized over every implementation."""
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory


@pytest.fixture
def repo(make_repo: RepoFactory) -> VectorStoreRepo:
    """The common case: chunks large enough that one document stays one fragment."""
    return make_repo(TextChunker(chunk_size=10_000))


@pytest.fixture
def chunking_repo(make_repo: RepoFactory) -> VectorStoreRepo:
    """Small chunks, so a long document is guaranteed to split into several fragments."""
    return make_repo(TextChunker(chunk_size=100, chunk_overlap=0))


LONG_TEXT = "A sentence about data structures. " * 50


async def test_search_on_empty_store_returns_empty(repo: VectorStoreRepo) -> None:
    assert await repo.search("anything", owner_id="u1") == []


async def test_add_then_search_returns_real_document_id_from_metadata(
    repo: VectorStoreRepo,
) -> None:
    await repo.add_documents(
        [
            Document(
                id="report.pdf",
                content="Content about quicksort",
                metadata={"filename": "report.pdf", "page": 1},
            )
        ],
        owner_id="u1",
    )

    results = await repo.search("quicksort", owner_id="u1", top_k=4)

    assert len(results) == 1
    assert results[0].id == "report.pdf"  # not "unknown"
    assert results[0].metadata["page"] == 1


async def test_add_documents_chunks_long_content(chunking_repo: VectorStoreRepo) -> None:
    await chunking_repo.add_documents(
        [Document(id="d.txt", content=LONG_TEXT, metadata={"filename": "d.txt"})], owner_id="u1"
    )

    # More than one result -> chunking actually happened.
    results = await chunking_repo.search("data structures", owner_id="u1", top_k=100)
    assert len(results) > 1


async def test_delete_by_document_id_removes_only_that_document(repo: VectorStoreRepo) -> None:
    await repo.add_documents(
        [Document(id="a.txt", content="alpha", metadata={"filename": "a.txt"})], owner_id="u1"
    )
    await repo.add_documents(
        [Document(id="b.txt", content="beta", metadata={"filename": "b.txt"})], owner_id="u1"
    )

    await repo.delete_by_document_id("a.txt", owner_id="u1")

    remaining_ids = {doc.id for doc in await repo.search("anything", owner_id="u1", top_k=100)}
    assert "a.txt" not in remaining_ids
    assert "b.txt" in remaining_ids


async def test_delete_on_empty_store_is_noop(repo: VectorStoreRepo) -> None:
    await repo.delete_by_document_id("does-not-exist", owner_id="u1")  # does not raise


async def test_reupload_replaces_chunks_without_orphans(chunking_repo: VectorStoreRepo) -> None:
    # A re-upload with fewer fragments must not leave the old (orphaned) ones behind.
    await chunking_repo.add_documents(
        [Document(id="d.txt", content=LONG_TEXT, metadata={"filename": "d.txt"})], owner_id="u1"
    )
    chunk_count_before = len(await chunking_repo.search("structures", owner_id="u1", top_k=1000))
    assert chunk_count_before > 1

    # Re-uploading the same document with shorter content → 1 fragment.
    await chunking_repo.add_documents(
        [Document(id="d.txt", content="short", metadata={"filename": "d.txt"})], owner_id="u1"
    )

    remaining = await chunking_repo.search("short structures", owner_id="u1", top_k=1000)
    assert len(remaining) == 1
    assert remaining[0].content == "short"


async def test_search_isolates_documents_by_owner(repo: VectorStoreRepo) -> None:
    # Retrieval isolation: a user's query never returns someone else's fragments.
    await repo.add_documents(
        [
            Document(
                id="secret.pdf",
                content="alice's secret about quicksort",
                metadata={"filename": "secret.pdf"},
            )
        ],
        owner_id="alice",
    )
    await repo.add_documents(
        [
            Document(
                id="bob.pdf",
                content="bob's notes about quicksort",
                metadata={"filename": "bob.pdf"},
            )
        ],
        owner_id="bob",
    )

    alice_results = {doc.id for doc in await repo.search("quicksort", owner_id="alice", top_k=100)}
    assert alice_results == {"secret.pdf"}
    assert "bob.pdf" not in alice_results
