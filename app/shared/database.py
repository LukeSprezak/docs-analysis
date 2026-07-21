"""Współdzielony asynchroniczny silnik bazy + pula połączeń (SQLAlchemy + psycopg3).

Zastępuje wzorzec „połączenie per-wywołanie" (`psycopg.connect(...)` w każdej metodzie
repo) jedną pulą na proces. Repozytoria pobierają połączenie z puli przez `db_connection()`,
zamiast otwierać i zamykać TCP/uwierzytelnianie przy każdym zapytaniu.

Silnik jest asynchroniczny (`postgresql+psycopg://` → psycopg3 async), więc zapytania
do bazy nie blokują event loopu FastAPI.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.shared.config import settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Leniwy singleton asynchronicznego silnika z pulą połączeń.

    `pool_pre_ping` odrzuca martwe połączenia (np. po restarcie bazy) zamiast oddawać je
    z puli; `pool_recycle` zamyka połączenia starsze niż 30 min (ochrona przed zerwaniem
    idle przez serwer/proxy).
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
    """Połączenie z puli w transakcji — commit na wyjściu, rollback przy wyjątku.

    `engine.begin()` otwiera transakcję i zatwierdza ją po czystym wyjściu z bloku
    (a wycofuje przy wyjątku), więc repozytoria nie muszą ręcznie wołać commit/rollback.
    """
    engine = get_engine()
    async with engine.begin() as connection:
        yield connection


async def dispose_engine() -> None:
    """Zamyka pulę (wywoływane w lifespanie FastAPI przy zamykaniu aplikacji)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None