"""Contract suite for the `SummaryRepo` port.

Every adapter runs the *same* assertions, exactly as in the `VectorStoreRepo` suite: a new
backing store is only finished once it is added to `ADAPTERS` below and passes unchanged.

The behaviours pinned here are the ones the summarize use cases depend on:

* `save` mints the id — the caller does not supply one — and reports it back both as the
  return value and on the summary it was handed, together with `created_at`;
* a summary survives the round trip — text and the whole `document_ids` list;
* every `save` is a new summary, never an update of an earlier one;
* reads are isolated per `owner_id` — `get_by_id` and `list_all` never reach another user;
* `list_all` returns the newest summary first and paginates over that order;
* `delete` removes one summary, only for its owner, and deleting nothing is a no-op.

There is one adapter today (`postgres`), so the whole suite is marked `integration` and
excluded from the default run (see `addopts` in pyproject.toml). Run it with a live database:

    POSTGRES_HOST=localhost POSTGRES_PORT=5433 uv run pytest -m integration \\
        tests/contracts/test_summary_repo_contract.py
"""

import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from sqlalchemy import text

from app.knowledge_management.domain.models import Summary
from app.knowledge_management.domain.repositories import SummaryRepo
from app.knowledge_management.infrastructure.persistence.postgres_summary_repo import (
    PostgresSummaryRepo,
)
from app.shared.database import db_connection, dispose_engine

RepoFactory = Callable[[], SummaryRepo]


@pytest.fixture
def owner_prefix() -> str:
    """Owner ids unique to one test.

    The summaries table is shared, so isolation cannot be a throwaway collection the way it
    is for the vector store. Instead every test writes under its own owner prefix, which
    also tells teardown exactly which rows to remove.
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
    yield PostgresSummaryRepo

    async with db_connection() as connection:
        await connection.execute(
            text("DELETE FROM summaries WHERE owner_id LIKE :prefix"),
            {"prefix": f"{owner_prefix}%"},
        )
    await dispose_engine()


ADAPTERS = [
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def repo(request: pytest.FixtureRequest) -> SummaryRepo:
    """The adapter under test, parametrized over every implementation."""
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory()


def _summary(text_: str = "The corpus is about sorting.") -> Summary:
    """A summary as the use case builds it — no id yet, the store assigns one."""
    return Summary(text=text_, document_ids=["doc-a", "doc-b"])


async def test_get_by_id_on_empty_store_returns_none(repo: SummaryRepo, owner: str) -> None:
    assert await repo.get_by_id(str(uuid.uuid4()), owner_id=owner) is None


async def test_save_assigns_the_id_and_reports_it_on_the_summary(
    repo: SummaryRepo, owner: str
) -> None:
    summary = _summary()

    summary_id = await repo.save(summary, owner_id=owner)

    assert summary_id
    assert summary.id == summary_id
    assert summary.created_at is not None


async def test_save_then_get_by_id_round_trips_text_and_document_ids(
    repo: SummaryRepo, owner: str
) -> None:
    summary = Summary(text="Three papers on retrieval.", document_ids=["a.pdf", "b.pdf", "c.pdf"])

    summary_id = await repo.save(summary, owner_id=owner)
    stored = await repo.get_by_id(summary_id, owner_id=owner)

    assert stored is not None
    assert stored.id == summary_id
    assert stored.text == "Three papers on retrieval."
    assert stored.document_ids == ["a.pdf", "b.pdf", "c.pdf"]
    assert stored.created_at is not None


async def test_every_save_creates_a_new_summary(repo: SummaryRepo, owner: str) -> None:
    """A summary is a record of one run, not a mutable document.

    Saving the same object twice must produce two summaries — the id carried on the object
    from the first save is an output, never a key the second save writes into.
    """
    summary = _summary()
    first_id = await repo.save(summary, owner_id=owner)
    second_id = await repo.save(summary, owner_id=owner)

    assert first_id != second_id
    assert len(await repo.list_all(owner_id=owner)) == 2


async def test_get_by_id_does_not_reach_another_owners_summary(
    repo: SummaryRepo, owner: str, intruder: str
) -> None:
    summary_id = await repo.save(_summary(), owner_id=owner)

    assert await repo.get_by_id(summary_id, owner_id=intruder) is None


async def test_list_all_returns_only_the_owners_summaries(
    repo: SummaryRepo, owner: str, intruder: str
) -> None:
    mine = await repo.save(_summary("Mine"), owner_id=owner)
    await repo.save(_summary("Theirs"), owner_id=intruder)

    assert [s.id for s in await repo.list_all(owner_id=owner)] == [mine]


async def test_list_all_returns_the_newest_summary_first(repo: SummaryRepo, owner: str) -> None:
    await repo.save(_summary("Older"), owner_id=owner)
    await repo.save(_summary("Newer"), owner_id=owner)

    assert [s.text for s in await repo.list_all(owner_id=owner)] == ["Newer", "Older"]


async def test_list_all_paginates_over_that_order(repo: SummaryRepo, owner: str) -> None:
    for text_ in ("first", "second", "third"):
        await repo.save(_summary(text_), owner_id=owner)

    page_one = await repo.list_all(owner_id=owner, limit=2)
    page_two = await repo.list_all(owner_id=owner, limit=2, offset=2)

    assert [s.text for s in page_one] == ["third", "second"]
    assert [s.text for s in page_two] == ["first"]


async def test_delete_removes_only_that_summary(repo: SummaryRepo, owner: str) -> None:
    kept = await repo.save(_summary("Kept"), owner_id=owner)
    removed = await repo.save(_summary("Removed"), owner_id=owner)

    await repo.delete(removed, owner_id=owner)

    assert [s.id for s in await repo.list_all(owner_id=owner)] == [kept]


async def test_delete_does_not_reach_another_owners_summary(
    repo: SummaryRepo, owner: str, intruder: str
) -> None:
    summary_id = await repo.save(_summary(), owner_id=owner)

    await repo.delete(summary_id, owner_id=intruder)

    assert await repo.get_by_id(summary_id, owner_id=owner) is not None


async def test_delete_of_an_unknown_summary_is_a_no_op(repo: SummaryRepo, owner: str) -> None:
    await repo.delete(str(uuid.uuid4()), owner_id=owner)
