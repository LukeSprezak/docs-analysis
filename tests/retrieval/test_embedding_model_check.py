"""The vector stores refuse to embed with a model other than the one their index was built with.

Vectors from two models are not comparable, so a mismatch has to fail loudly instead of
returning nonsense. Runs against every store that records the model — FAISS lives in process
memory only, so its model cannot change under it. Each test builds several repos over the
*same* collection / index, one per configured model, the way a redeploy with a changed
`LLM_PROVIDER` would.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from app.retrieval.chunker import TextChunker
from app.retrieval.ports import VectorStoreRepo
from app.retrieval.vector_store.neo4j import Neo4jVectorStoreRepo
from app.retrieval.vector_store.postgres import PostgresVectorStoreRepo
from app.shared.database import db_connection, dispose_engine
from app.shared.exceptions import EmbeddingModelMismatchException
from app.shared.kernel.document import Document
from tests.retrieval.test_vector_store_repo_contract import (
    NEO4J_TEST_PASSWORD,
    NEO4J_TEST_URI,
    NEO4J_TEST_USERNAME,
)

pytestmark = pytest.mark.integration

OWNER = "owner"
DOCUMENTS = [Document(id=f"{OWNER}::a.txt", content="Quicksort is fast.", metadata={})]


@dataclass
class StoreUnderTest:
    build: Callable[[str], VectorStoreRepo]
    """A repo over the shared collection / index, configured with the given model."""
    forget_recorded_model: Callable[[], Awaitable[None]]
    """Turns the index into one built before models were recorded."""


def _chunker() -> TextChunker:
    return TextChunker(chunk_size=10_000)


@pytest.fixture
async def postgres_store() -> AsyncIterator[StoreUnderTest]:
    collection_name = f"contract_test_{uuid.uuid4().hex}"
    created: list[PostgresVectorStoreRepo] = []

    def build(embedding_model: str) -> VectorStoreRepo:
        repo = PostgresVectorStoreRepo(
            embeddings=DeterministicFakeEmbedding(size=16),
            embedding_model=embedding_model,
            collection_name=collection_name,
            chunker=_chunker(),
        )
        created.append(repo)
        return repo

    async def forget_recorded_model() -> None:
        async with db_connection() as connection:
            await connection.execute(
                text("UPDATE langchain_pg_collection SET cmetadata = NULL WHERE name = :name"),
                {"name": collection_name},
            )

    yield StoreUnderTest(build, forget_recorded_model)

    await created[0].vector_store.adelete_collection()
    await dispose_engine()


@pytest.fixture
async def neo4j_store() -> AsyncIterator[StoreUnderTest]:
    suffix = uuid.uuid4().hex
    index_name = f"contract_test_{suffix}"
    created: list[Neo4jVectorStoreRepo] = []

    def build(embedding_model: str) -> VectorStoreRepo:
        repo = Neo4jVectorStoreRepo(
            embeddings=DeterministicFakeEmbedding(size=16),
            embedding_model=embedding_model,
            url=NEO4J_TEST_URI,
            username=NEO4J_TEST_USERNAME,
            password=NEO4J_TEST_PASSWORD,
            index_name=index_name,
            keyword_index_name=f"contract_test_kw_{suffix}",
            node_label=f"contract_test_{suffix}",
            chunker=_chunker(),
        )
        created.append(repo)
        return repo

    async def forget_recorded_model() -> None:
        created[0].vector_store.query(
            "MATCH (model:EmbeddingModel {index_name: $index_name}) DELETE model",
            params={"index_name": index_name},
        )

    yield StoreUnderTest(build, forget_recorded_model)

    created[0].vector_store.query(f"MATCH (n:`contract_test_{suffix}`) DETACH DELETE n")
    created[0].vector_store.query(
        "MATCH (model:EmbeddingModel {index_name: $index_name}) DELETE model",
        params={"index_name": index_name},
    )
    created[0].vector_store.query(f"DROP INDEX {index_name} IF EXISTS")
    for repo in created:
        await repo.close()


@pytest.fixture(params=["postgres_store", "neo4j_store"], ids=["postgres", "neo4j"])
def store(request: pytest.FixtureRequest) -> StoreUnderTest:
    store_under_test: StoreUnderTest = request.getfixturevalue(request.param)
    return store_under_test


async def test_same_model_is_accepted(store: StoreUnderTest) -> None:
    await store.build("model-a").add_documents(DOCUMENTS, OWNER)

    assert await store.build("model-a").search("quicksort", OWNER) != []


async def test_other_model_is_refused_on_search_and_upload(store: StoreUnderTest) -> None:
    await store.build("model-a").add_documents(DOCUMENTS, OWNER)

    with pytest.raises(EmbeddingModelMismatchException):
        await store.build("model-b").search("quicksort", OWNER)
    with pytest.raises(EmbeddingModelMismatchException):
        await store.build("model-b").add_documents(DOCUMENTS, OWNER)


async def test_index_without_a_recorded_model_adopts_the_configured_one(
    store: StoreUnderTest,
) -> None:
    await store.build("model-a").add_documents(DOCUMENTS, OWNER)
    await store.forget_recorded_model()

    await store.build("model-b").search("quicksort", OWNER)

    # Adopted, not ignored: from now on the index belongs to model-b.
    with pytest.raises(EmbeddingModelMismatchException):
        await store.build("model-a").search("quicksort", OWNER)


async def test_emptied_index_adopts_a_new_model(store: StoreUnderTest) -> None:
    # "Delete everything and upload again" is the documented way to switch models.
    await store.build("model-a").add_documents(DOCUMENTS, OWNER)
    await store.build("model-a").delete_by_document_id(DOCUMENTS[0].id, OWNER)

    await store.build("model-b").add_documents(DOCUMENTS, OWNER)

    with pytest.raises(EmbeddingModelMismatchException):
        await store.build("model-a").search("quicksort", OWNER)
