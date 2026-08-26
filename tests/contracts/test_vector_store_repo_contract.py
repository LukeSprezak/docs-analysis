"""Contract suite for the `VectorStoreRepo` port.

Every adapter runs the *same* assertions. This is what makes the backing store genuinely
swappable rather than swappable on paper: a new adapter (Neo4j, Qdrant, …) is only finished
once it is added to `ADAPTERS` below and passes unchanged.

The behaviours pinned here are the ones the use cases and the security model depend on:

* retrieval is isolated per `owner_id` — a query never reaches another user's chunks;
* so is deletion — removing a document never touches another user's chunks;
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
from app.knowledge_management.infrastructure.persistence.neo4j_vectorstore_repo import (
    Neo4jVectorStoreRepo,
)
from app.knowledge_management.infrastructure.persistence.postgres_vectorstore_repo import (
    PostgresVectorStoreRepo,
)
from app.knowledge_management.infrastructure.text.text_chunker import TextChunker
from app.shared.config import settings
from app.shared.database import dispose_engine

# A repo builder takes the chunker the test wants (chunk size decides how many fragments a
# document produces) plus whether hybrid retrieval is on, and returns a ready adapter.
RepoFactory = Callable[[TextChunker, bool], VectorStoreRepo]

# Small, deterministic vectors — no network, stable ordering between runs.
EMBEDDING_SIZE = 16

# Server-backed adapters read their connection details from the same settings the app uses.
# The committed .env points at the docker-compose service names, so running these from the
# host means overriding the hostnames, e.g.
#   POSTGRES_HOST=localhost POSTGRES_PORT=5433 \
#   NEO4J_URI=bolt://localhost:7687 pytest -m integration
NEO4J_TEST_URI = settings.NEO4J_URI or "bolt://localhost:7687"
NEO4J_TEST_USERNAME = settings.NEO4J_USERNAME or "neo4j"
NEO4J_TEST_PASSWORD = settings.NEO4J_PASSWORD or ""


def _embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


@pytest.fixture
def faiss_factory() -> RepoFactory:
    def build(chunker: TextChunker, enable_hybrid_search: bool) -> VectorStoreRepo:
        if enable_hybrid_search:
            # The in-memory store has no keyword index, so it is absent from HYBRID_ADAPTERS.
            raise NotImplementedError("FAISS does not support hybrid retrieval")
        return FaissVectorStoreRepo(embeddings=_embeddings(), chunker=chunker)

    return build


@pytest.fixture
async def postgres_factory() -> AsyncIterator[RepoFactory]:
    """Postgres adapter against a live database, isolated in a throwaway collection.

    Each test gets its own `collection_name`, so repeated runs never see each other's
    vectors. Teardown drops the collections and disposes the shared engine — the pool is a
    module-level singleton bound to the loop that created it, and pytest-asyncio gives each
    test a fresh loop.
    """
    created: list[PostgresVectorStoreRepo] = []

    def build(chunker: TextChunker, enable_hybrid_search: bool) -> VectorStoreRepo:
        repo = PostgresVectorStoreRepo(
            embeddings=_embeddings(),
            collection_name=f"contract_test_{uuid.uuid4().hex}",
            chunker=chunker,
            enable_hybrid_search=enable_hybrid_search,
        )
        created.append(repo)
        return repo

    yield build

    for repo in created:
        await repo.vector_store.adelete_collection()
    await dispose_engine()


@pytest.fixture
async def neo4j_factory() -> AsyncIterator[RepoFactory]:
    """Neo4j adapter against a live server, isolated by a per-test node label.

    Neo4j has no notion of a collection, so isolation is the node label: each test writes to
    its own `:contract_test_<uuid>` label with its own vector index, and teardown deletes
    both. Sharing one label would leak vectors between tests through the shared index.
    """
    created: list[Neo4jVectorStoreRepo] = []

    def build(chunker: TextChunker, enable_hybrid_search: bool) -> VectorStoreRepo:
        suffix = uuid.uuid4().hex
        repo = Neo4jVectorStoreRepo(
            embeddings=_embeddings(),
            url=NEO4J_TEST_URI,
            username=NEO4J_TEST_USERNAME,
            password=NEO4J_TEST_PASSWORD,
            index_name=f"contract_test_{suffix}",
            keyword_index_name=f"contract_test_kw_{suffix}",
            node_label=f"contract_test_{suffix}",
            chunker=chunker,
            enable_hybrid_search=enable_hybrid_search,
        )
        created.append(repo)
        return repo

    yield build

    for repo in created:
        repo.vector_store.query(f"MATCH (n:`{repo.node_label}`) DETACH DELETE n")
        repo.vector_store.query(f"DROP INDEX {repo.vector_store.index_name} IF EXISTS")
        await repo.close()


ADAPTERS = [
    pytest.param("faiss_factory", id="faiss"),
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
    pytest.param("neo4j_factory", id="neo4j", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def make_repo(request: pytest.FixtureRequest) -> RepoFactory:
    """The adapter under test, as a builder parametrized over every implementation."""
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory


@pytest.fixture
def repo(make_repo: RepoFactory) -> VectorStoreRepo:
    """The common case: chunks large enough that one document stays one fragment."""
    return make_repo(TextChunker(chunk_size=10_000), False)


@pytest.fixture
def chunking_repo(make_repo: RepoFactory) -> VectorStoreRepo:
    """Small chunks, so a long document is guaranteed to split into several fragments."""
    return make_repo(TextChunker(chunk_size=100, chunk_overlap=0), False)


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


async def test_delete_isolates_documents_by_owner(repo: VectorStoreRepo) -> None:
    # Write isolation, symmetric to the read case above: deleting a document must not touch
    # another user's fragments that happen to share the id. Production ids are namespaced per
    # user, but that is an invariant of the caller — the port promises the owner filter here.
    await repo.add_documents(
        [
            Document(
                id="shared.pdf",
                content="alice's notes about quicksort",
                metadata={"filename": "shared.pdf"},
            )
        ],
        owner_id="alice",
    )
    await repo.add_documents(
        [
            Document(
                id="shared.pdf",
                content="bob's notes about quicksort",
                metadata={"filename": "shared.pdf"},
            )
        ],
        owner_id="bob",
    )

    await repo.delete_by_document_id("shared.pdf", owner_id="alice")

    bob_results = await repo.search("quicksort", owner_id="bob", top_k=100)
    assert {doc.id for doc in bob_results} == {"shared.pdf"}
    assert await repo.search("quicksort", owner_id="alice", top_k=100) == []


# --- Hybrid retrieval -------------------------------------------------------------------
# Only adapters with a keyword index alongside the vectors. The embeddings here are random
# (deterministic, but semantically meaningless), so a hit on a distinctive rare term is
# evidence the keyword half of the fusion ran — a vector-only search could not find it.

HYBRID_ADAPTERS = [
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
    pytest.param("neo4j_factory", id="neo4j", marks=pytest.mark.integration),
]

RARE_TERM = "zzyzx"


@pytest.fixture(params=HYBRID_ADAPTERS)
def hybrid_repo(request: pytest.FixtureRequest) -> VectorStoreRepo:
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory(TextChunker(chunk_size=10_000), True)


async def test_hybrid_search_finds_documents_by_keyword(hybrid_repo: VectorStoreRepo) -> None:
    await hybrid_repo.add_documents(
        [Document(id="rare.txt", content=f"The {RARE_TERM} protocol", metadata={})],
        owner_id="u1",
    )
    # Enough noise that a vector-only search cannot plausibly surface the right document in
    # two slots: the embeddings are random, so this asserts the keyword branch really ran.
    # (Verified by flipping the fixture to vector-only, where this test fails.)
    for filler in range(30):
        await hybrid_repo.add_documents(
            [Document(id=f"filler{filler}.txt", content=f"unrelated prose {filler}", metadata={})],
            owner_id="u1",
        )

    results = await hybrid_repo.search(RARE_TERM, owner_id="u1", top_k=2)

    assert "rare.txt" in {doc.id for doc in results}


async def test_hybrid_search_isolates_documents_by_owner(hybrid_repo: VectorStoreRepo) -> None:
    # The owner filter has to hold on the keyword branch too — that branch is hand-written
    # SQL/Cypher in both adapters, and a missing WHERE there would leak across users
    # regardless of how well the vector branch is filtered.
    await hybrid_repo.add_documents(
        [Document(id="alice.txt", content=f"alice's {RARE_TERM} notes", metadata={})],
        owner_id="alice",
    )
    await hybrid_repo.add_documents(
        [Document(id="bob.txt", content=f"bob's {RARE_TERM} notes", metadata={})],
        owner_id="bob",
    )

    alice_results = {doc.id for doc in await hybrid_repo.search(RARE_TERM, "alice", top_k=100)}

    assert alice_results == {"alice.txt"}
