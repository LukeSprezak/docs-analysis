"""Contract suite for the `ConversationRepo` port.

Every adapter runs the *same* assertions, exactly as in the `VectorStoreRepo` suite: a new
backing store is only finished once it is added to `ADAPTERS` below and passes unchanged.

The behaviours pinned here are the ones the chat use cases and the security model depend on:

* a conversation survives the round trip — id, title and every message field;
* saving the same id twice updates that conversation instead of creating a second one;
* **a save under a different `owner_id` never touches an existing conversation, and gains
  the caller nothing** — the conflict key is the owner *and* the id, not the id alone;
* reads are isolated per `owner_id` — `get_by_id` and `list_all` never reach another user;
* `list_all` returns the newest conversation first and paginates over that order;
* `delete` removes one conversation, only for its owner, and deleting nothing is a no-op.

There is one adapter today (`postgres`), so the whole suite is marked `integration` and
excluded from the default run (see `addopts` in pyproject.toml). Run it with a live database:

    POSTGRES_HOST=localhost POSTGRES_PORT=5433 uv run pytest -m integration \\
        tests/contracts/test_conversation_repo_contract.py
"""

import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from sqlalchemy import text

from app.knowledge_management.domain.models import ChatMessage, Conversation
from app.knowledge_management.domain.repositories import ConversationRepo
from app.knowledge_management.infrastructure.persistence.postgres_conversation_repo import (
    PostgresConversationRepo,
)
from app.shared.database import db_connection, dispose_engine

RepoFactory = Callable[[], ConversationRepo]


@pytest.fixture
def owner_prefix() -> str:
    """Owner ids unique to one test.

    The conversations table is shared, so isolation cannot be a throwaway collection the way
    it is for the vector store. Instead every test writes under its own owner prefix, which
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
    yield PostgresConversationRepo

    async with db_connection() as connection:
        await connection.execute(
            text("DELETE FROM conversations WHERE owner_id LIKE :prefix"),
            {"prefix": f"{owner_prefix}%"},
        )
    await dispose_engine()


ADAPTERS = [
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def repo(request: pytest.FixtureRequest) -> ConversationRepo:
    """The adapter under test, parametrized over every implementation."""
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory()


def _conversation(
    title: str = "Retrieval notes", messages: list[ChatMessage] | None = None
) -> Conversation:
    """A conversation with a server-generated id, as the use cases build it."""
    return Conversation(
        id=str(uuid.uuid4()),
        title=title,
        messages=messages if messages is not None else [ChatMessage(role="user", content="hello")],
    )


async def test_get_by_id_on_empty_store_returns_none(repo: ConversationRepo, owner: str) -> None:
    assert await repo.get_by_id(str(uuid.uuid4()), owner_id=owner) is None


async def test_save_then_get_by_id_round_trips_every_message_field(
    repo: ConversationRepo, owner: str
) -> None:
    conversation = _conversation(
        title="Quicksort",
        messages=[
            ChatMessage(
                role="user", content="how does it partition?", timestamp="2026-01-01T00:00:00"
            ),
            ChatMessage(role="assistant", content="around a pivot"),
        ],
    )

    await repo.save(conversation, owner_id=owner)
    stored = await repo.get_by_id(conversation.id, owner_id=owner)

    assert stored is not None
    assert stored.id == conversation.id
    assert stored.title == "Quicksort"
    assert [(m.role, m.content, m.timestamp) for m in stored.messages] == [
        ("user", "how does it partition?", "2026-01-01T00:00:00"),
        ("assistant", "around a pivot", None),
    ]


async def test_saving_the_same_id_twice_updates_in_place(
    repo: ConversationRepo, owner: str
) -> None:
    conversation = _conversation(title="First title")
    await repo.save(conversation, owner_id=owner)

    conversation.title = "Second title"
    conversation.messages.append(ChatMessage(role="assistant", content="reply"))
    await repo.save(conversation, owner_id=owner)

    stored = await repo.get_by_id(conversation.id, owner_id=owner)
    assert stored is not None
    assert stored.title == "Second title"
    assert len(stored.messages) == 2
    assert len(await repo.list_all(owner_id=owner)) == 1


async def test_get_by_id_does_not_reach_another_owners_conversation(
    repo: ConversationRepo, owner: str, intruder: str
) -> None:
    conversation = _conversation()
    await repo.save(conversation, owner_id=owner)

    assert await repo.get_by_id(conversation.id, owner_id=intruder) is None


async def test_save_under_another_owner_leaves_the_conversation_untouched(
    repo: ConversationRepo, owner: str, intruder: str
) -> None:
    """SEC-01: knowing a conversation id must not be enough to overwrite it.

    The id is not a secret — it comes back in `ChatResponse`, in the listing, in logs and
    URLs. An upsert keyed on the id alone lets anyone holding it replace the title, wipe the
    history and take over ownership.
    """
    victim = _conversation(
        title="Owner's notes", messages=[ChatMessage(role="user", content="original")]
    )
    await repo.save(victim, owner_id=owner)

    hijack = Conversation(
        id=victim.id,
        title="Hijacked",
        messages=[ChatMessage(role="user", content="attacker text")],
    )
    await repo.save(hijack, owner_id=intruder)

    survived = await repo.get_by_id(victim.id, owner_id=owner)
    assert survived is not None, "a foreign save deleted the owner's conversation"
    assert survived.title == "Owner's notes"
    assert [m.content for m in survived.messages] == ["original"]


async def test_save_under_another_owner_creates_nothing_for_that_owner(
    repo: ConversationRepo, owner: str, intruder: str
) -> None:
    """The other half of SEC-01: the rejected write must not land anywhere either.

    One id means one conversation. A save that loses the conflict is dropped — it does not
    become a second row the intruder owns.
    """
    victim = _conversation()
    await repo.save(victim, owner_id=owner)

    await repo.save(Conversation(id=victim.id, title="Hijacked", messages=[]), owner_id=intruder)

    assert await repo.get_by_id(victim.id, owner_id=intruder) is None
    assert await repo.list_all(owner_id=intruder) == []


async def test_list_all_returns_only_the_owners_conversations(
    repo: ConversationRepo, owner: str, intruder: str
) -> None:
    mine = _conversation(title="Mine")
    theirs = _conversation(title="Theirs")
    await repo.save(mine, owner_id=owner)
    await repo.save(theirs, owner_id=intruder)

    assert [c.id for c in await repo.list_all(owner_id=owner)] == [mine.id]


async def test_list_all_returns_the_newest_conversation_first(
    repo: ConversationRepo, owner: str
) -> None:
    older = _conversation(title="Older")
    newer = _conversation(title="Newer")
    await repo.save(older, owner_id=owner)
    await repo.save(newer, owner_id=owner)

    assert [c.title for c in await repo.list_all(owner_id=owner)] == ["Newer", "Older"]


async def test_list_all_paginates_over_that_order(repo: ConversationRepo, owner: str) -> None:
    for title in ("first", "second", "third"):
        await repo.save(_conversation(title=title), owner_id=owner)

    page_one = await repo.list_all(owner_id=owner, limit=2)
    page_two = await repo.list_all(owner_id=owner, limit=2, offset=2)

    assert [c.title for c in page_one] == ["third", "second"]
    assert [c.title for c in page_two] == ["first"]


async def test_delete_removes_only_that_conversation(repo: ConversationRepo, owner: str) -> None:
    kept = _conversation(title="Kept")
    removed = _conversation(title="Removed")
    await repo.save(kept, owner_id=owner)
    await repo.save(removed, owner_id=owner)

    await repo.delete(removed.id, owner_id=owner)

    assert [c.id for c in await repo.list_all(owner_id=owner)] == [kept.id]


async def test_delete_does_not_reach_another_owners_conversation(
    repo: ConversationRepo, owner: str, intruder: str
) -> None:
    conversation = _conversation()
    await repo.save(conversation, owner_id=owner)

    await repo.delete(conversation.id, owner_id=intruder)

    assert await repo.get_by_id(conversation.id, owner_id=owner) is not None


async def test_delete_of_an_unknown_conversation_is_a_no_op(
    repo: ConversationRepo, owner: str
) -> None:
    await repo.delete(str(uuid.uuid4()), owner_id=owner)
