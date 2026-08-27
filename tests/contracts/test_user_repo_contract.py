"""Contract suite for the `UserRepo` port.

Every adapter runs the *same* assertions, exactly as in the `VectorStoreRepo` suite: a new
backing store is only finished once it is added to `ADAPTERS` below and passes unchanged.

The behaviours pinned here are the ones registration and login depend on:

* a user survives the round trip — id, e-mail, password hash and the stored `created_at`;
* both lookups reach the same user: by e-mail (login) and by id (the token subject);
* saving the same id twice updates the credentials in place instead of creating a second
  account;
* **a save carrying somebody else's id never touches their account, and gains the caller
  nothing** — the e-mail guards the upsert here the way `owner_id` guards it elsewhere
  (SEC-02).

There is one adapter today (`postgres`), so the whole suite is marked `integration` and
excluded from the default run (see `addopts` in pyproject.toml). Run it with a live database:

    POSTGRES_HOST=localhost POSTGRES_PORT=5433 uv run pytest -m integration \\
        tests/contracts/test_user_repo_contract.py
"""

import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from sqlalchemy import text

from app.identity.domain.models import User
from app.identity.domain.repositories import UserRepo
from app.identity.infrastructure.postgres_user_repo import PostgresUserRepo
from app.shared.database import db_connection, dispose_engine

RepoFactory = Callable[[], UserRepo]


@pytest.fixture
def email_prefix() -> str:
    """E-mails unique to one test.

    A user has no `owner_id` to scope it — the e-mail is what identifies the account, so it
    carries the per-test prefix instead and tells teardown which rows to remove.
    """
    return f"contract_test_{uuid.uuid4().hex}"


@pytest.fixture
def email(email_prefix: str) -> str:
    return f"{email_prefix}_owner@example.test"


@pytest.fixture
def other_email(email_prefix: str) -> str:
    return f"{email_prefix}_intruder@example.test"


@pytest.fixture
async def postgres_factory(email_prefix: str) -> AsyncIterator[RepoFactory]:
    """Postgres adapter against a live database.

    Teardown deletes this test's rows and disposes the shared engine — the pool is a
    module-level singleton bound to the loop that created it, and pytest-asyncio gives each
    test a fresh loop.
    """
    yield PostgresUserRepo

    async with db_connection() as connection:
        await connection.execute(
            text("DELETE FROM users WHERE email LIKE :prefix"),
            {"prefix": f"{email_prefix}%"},
        )
    await dispose_engine()


ADAPTERS = [
    pytest.param("postgres_factory", id="postgres", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def repo(request: pytest.FixtureRequest) -> UserRepo:
    """The adapter under test, parametrized over every implementation."""
    factory: RepoFactory = request.getfixturevalue(request.param)
    return factory()


PLACEHOLDER_HASH = "$2b$12$hash"


def _user(email: str, hashed_password: str = PLACEHOLDER_HASH) -> User:
    """A user with a client-generated id, as the registration use case builds it."""
    return User(id=str(uuid.uuid4()), email=email, hashed_password=hashed_password)


async def test_get_by_email_on_empty_store_returns_none(repo: UserRepo, email: str) -> None:
    assert await repo.get_by_email(email) is None


async def test_get_by_id_on_empty_store_returns_none(repo: UserRepo) -> None:
    assert await repo.get_by_id(str(uuid.uuid4())) is None


async def test_save_then_get_by_email_round_trips_every_field(repo: UserRepo, email: str) -> None:
    user = _user(email, hashed_password="$2b$12$originalhash")

    await repo.save(user)
    stored = await repo.get_by_email(email)

    assert stored is not None
    assert stored.id == user.id
    assert stored.email == email
    assert stored.hashed_password == "$2b$12$originalhash"
    assert stored.created_at is not None


async def test_get_by_id_reaches_the_same_user(repo: UserRepo, email: str) -> None:
    user = _user(email)
    await repo.save(user)

    stored = await repo.get_by_id(user.id)

    assert stored is not None
    assert stored.email == email


async def test_saving_the_same_id_twice_updates_the_credentials_in_place(
    repo: UserRepo, email: str
) -> None:
    user = _user(email, hashed_password="$2b$12$firsthash")
    await repo.save(user)

    user.hashed_password = "$2b$12$secondhash"
    await repo.save(user)

    stored = await repo.get_by_id(user.id)
    assert stored is not None
    assert stored.hashed_password == "$2b$12$secondhash"


async def test_save_under_another_email_leaves_that_account_untouched(
    repo: UserRepo, email: str, other_email: str
) -> None:
    """SEC-02: holding somebody's user id must not be enough to take their account.

    The id is the token subject and travels in every response body, so it is not a secret.
    An upsert keyed on the id alone would let a registration carrying that id overwrite the
    victim's password hash.
    """
    victim = _user(email, hashed_password="$2b$12$victimhash")
    await repo.save(victim)

    await repo.save(User(id=victim.id, email=other_email, hashed_password="$2b$12$attackerhash"))

    survived = await repo.get_by_email(email)
    assert survived is not None, "a foreign save deleted the account"
    assert survived.hashed_password == "$2b$12$victimhash"


async def test_save_under_another_email_creates_nothing_for_that_email(
    repo: UserRepo, email: str, other_email: str
) -> None:
    """The other half of SEC-02: the rejected write must not land anywhere either.

    One id means one account. A save that loses the conflict is dropped — it does not
    become a second row under the attacker's e-mail.
    """
    victim = _user(email)
    await repo.save(victim)

    await repo.save(User(id=victim.id, email=other_email, hashed_password="$2b$12$attackerhash"))

    assert await repo.get_by_email(other_email) is None
