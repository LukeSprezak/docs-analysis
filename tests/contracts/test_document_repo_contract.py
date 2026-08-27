"""Contract suite for the `DocumentRepo` port.

Every adapter runs the *same* assertions, exactly as in the `VectorStoreRepo` suite: a new
backing store is only finished once it is added to `ADAPTERS` below and passes unchanged.

The behaviours pinned here are the ones the document use cases and the security model
depend on:

* a document survives the round trip — id, content and the whole metadata mapping;
* saving the same id twice updates that document instead of creating a second one;
* **a save under a different `owner_id` never touches an existing document, and gains the
  caller nothing** — the conflict key is the owner *and* the id, not the id alone (SEC-02);
* reads are isolated per `owner_id` — `get_by_id` and `list_all` never reach another user;
* `list_all` returns the newest document first and paginates over that order;
* `delete` removes one document, only for its owner, and deleting nothing is a no-op.

There is one adapter today (`postgres`), so the whole suite is marked `integration` and
excluded from the default run (see `addopts` in pyproject.toml). Run it with a live database:

    POSTGRES_HOST=localhost POSTGRES_PORT=5433 uv run pytest -m integration \\
        tests/contracts/test_document_repo_contract.py
"""

import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from sqlalchemy import text

from app.knowledge_management.domain.models import Document
from app.knowledge_management.domain.repositories import DocumentRepo
from app.knowledge_management.infrastructure.persistence.postgres_document_repo import (
    PostgresDocumentRepo,
)
from app.shared.database import db_connection, dispose_engine

RepoFactory = Callable[[], DocumentRepo]


@pytest.fixture
def owner_prefix() -> str:
    """Owner ids unique to one test.

    The documents table is shared, so isolation cannot be a throwaway collection the way it
    is for the vector store. Instead every test writes under its own owner prefix, which
    also tells teardown exactly which rows to remove — including rows whose `owner_id` a
    buggy adapter rewrote, since both sides of every hijack scenario share the prefix.
    """
    return f"contract_test_{uuid.uuid4().hex}"


@pytest.fixture
def owner(owner_prefix: str) -> str:
    return f"{owner_prefix}_owner"


@pytest.fixture
def intruder(owner_prefix: str) -> str:
    return f"{owner_prefix}_intruder"


@pytest.fixture
async def postgres_factory(owner_prefix: str) -> AsyncIterator[RepoFactory]:
    """Postgres adapter against a live database.

    Teardown deletes this test's rows and disposes the shared engine — the pool is a
    module-level singleton bound to the loop that created it, and pytest-asyncio gives each
    test a fresh loop.
    """
    yield PostgresDocumentRepo

    async with db_connection() as connection:
        await connection.execute(
            text("DELETE FROM documents WHERE owner_id LIKE :prefix"),
            {"prefix": f"{owner_prefix}%"},
        )
    await dispose_engine()


ADAPTERS = [
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def repo(request: pytest.FixtureRequest) -> DocumentRepo:
    """The adapter under test, parametrized over every implementation."""
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory()


def _document(
    owner: str, content: str = "chunk text", metadata: dict[str, object] | None = None
) -> Document:
    """A document under an id namespaced by owner, as the upload use case builds it."""
    return Document(
        id=f"{owner}::{uuid.uuid4().hex}.pdf",
        content=content,
        metadata=metadata if metadata is not None else {"filename": "notes.pdf"},
    )


async def test_get_by_id_on_empty_store_returns_none(repo: DocumentRepo, owner: str) -> None:
    assert await repo.get_by_id(f"{owner}::missing.pdf", owner_id=owner) is None


async def test_save_then_get_by_id_round_trips_content_and_metadata(
    repo: DocumentRepo, owner: str
) -> None:
    document = _document(
        owner,
        content="quicksort partitions around a pivot",
        metadata={"filename": "quicksort.pdf", "page": 3, "tags": ["sorting", "recursion"]},
    )

    await repo.save(document, owner_id=owner)
    stored = await repo.get_by_id(document.id, owner_id=owner)

    assert stored is not None
    assert stored.id == document.id
    assert stored.content == "quicksort partitions around a pivot"
    assert stored.metadata == {
        "filename": "quicksort.pdf",
        "page": 3,
        "tags": ["sorting", "recursion"],
    }


async def test_saving_the_same_id_twice_updates_in_place(repo: DocumentRepo, owner: str) -> None:
    document = _document(owner, content="first version")
    await repo.save(document, owner_id=owner)

    document.content = "second version"
    document.metadata = {"filename": "notes.pdf", "revision": 2}
    await repo.save(document, owner_id=owner)

    stored = await repo.get_by_id(document.id, owner_id=owner)
    assert stored is not None
    assert stored.content == "second version"
    assert stored.metadata == {"filename": "notes.pdf", "revision": 2}
    assert len(await repo.list_all(owner_id=owner)) == 1


async def test_get_by_id_does_not_reach_another_owners_document(
    repo: DocumentRepo, owner: str, intruder: str
) -> None:
    document = _document(owner)
    await repo.save(document, owner_id=owner)

    assert await repo.get_by_id(document.id, owner_id=intruder) is None


async def test_save_under_another_owner_leaves_the_document_untouched(
    repo: DocumentRepo, owner: str, intruder: str
) -> None:
    """SEC-02: knowing a document id must not be enough to overwrite it.

    The id travels in listings, citations and URLs, so it is not a secret. An upsert keyed
    on the id alone lets anyone holding it replace the content the owner uploaded.
    """
    victim = _document(owner, content="the owner's text")
    await repo.save(victim, owner_id=owner)

    hijack = Document(id=victim.id, content="attacker text", metadata={"filename": "evil.pdf"})
    await repo.save(hijack, owner_id=intruder)

    survived = await repo.get_by_id(victim.id, owner_id=owner)
    assert survived is not None, "a foreign save deleted the owner's document"
    assert survived.content == "the owner's text"
    assert survived.metadata == {"filename": "notes.pdf"}


async def test_save_under_another_owner_creates_nothing_for_that_owner(
    repo: DocumentRepo, owner: str, intruder: str
) -> None:
    """The other half of SEC-02: the rejected write must not land anywhere either.

    One id means one document. A save that loses the conflict is dropped — it does not
    become a second row the intruder owns.
    """
    victim = _document(owner)
    await repo.save(victim, owner_id=owner)

    await repo.save(Document(id=victim.id, content="attacker text", metadata={}), owner_id=intruder)

    assert await repo.get_by_id(victim.id, owner_id=intruder) is None
    assert await repo.list_all(owner_id=intruder) == []


async def test_list_all_returns_only_the_owners_documents(
    repo: DocumentRepo, owner: str, intruder: str
) -> None:
    mine = _document(owner)
    theirs = _document(intruder)
    await repo.save(mine, owner_id=owner)
    await repo.save(theirs, owner_id=intruder)

    assert [d.id for d in await repo.list_all(owner_id=owner)] == [mine.id]


async def test_list_all_returns_the_newest_document_first(repo: DocumentRepo, owner: str) -> None:
    older = _document(owner, content="older")
    newer = _document(owner, content="newer")
    await repo.save(older, owner_id=owner)
    await repo.save(newer, owner_id=owner)

    assert [d.content for d in await repo.list_all(owner_id=owner)] == ["newer", "older"]


async def test_list_all_paginates_over_that_order(repo: DocumentRepo, owner: str) -> None:
    for content in ("first", "second", "third"):
        await repo.save(_document(owner, content=content), owner_id=owner)

    page_one = await repo.list_all(owner_id=owner, limit=2)
    page_two = await repo.list_all(owner_id=owner, limit=2, offset=2)

    assert [d.content for d in page_one] == ["third", "second"]
    assert [d.content for d in page_two] == ["first"]


async def test_delete_removes_only_that_document(repo: DocumentRepo, owner: str) -> None:
    kept = _document(owner, content="kept")
    removed = _document(owner, content="removed")
    await repo.save(kept, owner_id=owner)
    await repo.save(removed, owner_id=owner)

    await repo.delete(removed.id, owner_id=owner)

    assert [d.id for d in await repo.list_all(owner_id=owner)] == [kept.id]


async def test_delete_does_not_reach_another_owners_document(
    repo: DocumentRepo, owner: str, intruder: str
) -> None:
    document = _document(owner)
    await repo.save(document, owner_id=owner)

    await repo.delete(document.id, owner_id=intruder)

    assert await repo.get_by_id(document.id, owner_id=owner) is not None


async def test_delete_of_an_unknown_document_is_a_no_op(repo: DocumentRepo, owner: str) -> None:
    await repo.delete(f"{owner}::missing.pdf", owner_id=owner)
