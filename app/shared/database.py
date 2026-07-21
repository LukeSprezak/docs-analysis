"""Shared async database engine + connection pool (SQLAlchemy + psycopg3).

Replaces the "connection per call" pattern (`psycopg.connect(...)` in every repo method)
with a single pool per process. Repositories take a connection from the pool via
`db_connection()` instead of opening and closing TCP/authentication on every query.

The engine is asynchronous (`postgresql+psycopg://` → psycopg3 async), so database queries
do not block the FastAPI event loop.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.shared.config import settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Lazy singleton of the async engine with its connection pool.

    `pool_pre_ping` discards dead connections (e.g. after a database restart) instead of
    handing them out of the pool; `pool_recycle` closes connections older than 30 min
    (protection against idle connections being dropped by the server/proxy).
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


@asynccontextmanager
async def db_connection() -> AsyncIterator[AsyncConnection]:
    """A pooled connection inside a transaction — commit on exit, rollback on exception.

    `engine.begin()` opens a transaction and commits it when the block exits cleanly (and
    rolls back on an exception), so repositories never call commit/rollback by hand.
    """
    engine = get_engine()
    async with engine.begin() as connection:
        yield connection


async def dispose_engine() -> None:
    """Closes the pool (called from the FastAPI lifespan on application shutdown)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
