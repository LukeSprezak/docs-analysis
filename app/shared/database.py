from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.shared.config import settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine | None:
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
    engine = get_engine()
    async with engine.begin() as connection:
        yield connection


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
