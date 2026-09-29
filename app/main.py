from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.conversations import dependencies as conversations_dependencies
from app.conversations.ui import router as conversations_router
from app.documents import dependencies as documents_dependencies
from app.documents.ui import router as documents_router
from app.identity import router as identity_router
from app.retrieval import dependencies as retrieval_dependencies
from app.retrieval.ui import router as retrieval_router
from app.shared import translations_router
from app.shared.config import settings
from app.shared.database import dispose_engine
from app.shared.exception_handlers import (
    app_exception_handler,
    global_exception_handler,
    rate_limit_exceeded_handler,
)
from app.shared.exceptions import AppException
from app.shared.logging import setup_logging
from app.shared.middleware import LoggingMiddleware
from app.shared.rate_limit import limiter
from app.summaries import router as summaries_router

setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Opens the repositories on startup and releases the connections on shutdown.

    Building them here rather than on whichever request arrives first means an adapter that
    connects in its constructor (Neo4j) fails the startup instead of a query, and that two
    concurrent first requests cannot each build one.

    On the way out repositories go first — an adapter that owns a driver has to close it
    itself — and the shared SQLAlchemy pool second, since the record repositories borrow from
    it. Without this the pool was never disposed: connections stayed open until the process
    died, which a reloading dev server or a test run does repeatedly.
    """
    documents_dependencies.init()
    retrieval_dependencies.init()
    conversations_dependencies.init()
    yield
    await documents_dependencies.shutdown()
    await retrieval_dependencies.shutdown()
    await conversations_dependencies.shutdown()
    await dispose_engine()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    # Never "*" together with credentials — the origin list is an allowlist from env.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(identity_router.router, prefix=settings.API_V1_STR)
app.include_router(documents_router.router, prefix=settings.API_V1_STR)
app.include_router(summaries_router.router, prefix=settings.API_V1_STR)
app.include_router(conversations_router.router, prefix=settings.API_V1_STR)
app.include_router(retrieval_router.router, prefix=settings.API_V1_STR)
app.include_router(translations_router.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
