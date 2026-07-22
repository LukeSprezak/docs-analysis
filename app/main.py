from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.identity.ui import auth
from app.knowledge_management.ui.api.routers import (
    chat,
    documents,
    qa,
    summarize,
    translations,
)
from app.shared.config import settings
from app.shared.database import dispose_engine
from app.shared.dependencies import shutdown_repositories
from app.shared.exception_handlers import (
    app_exception_handler,
    global_exception_handler,
    rate_limit_exceeded_handler,
)
from app.shared.exceptions import AppException
from app.shared.logging import setup_logging
from app.shared.middleware import LoggingMiddleware
from app.shared.rate_limit import limiter

setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Releases connections on shutdown.

    Repositories go first — an adapter that owns a driver (Neo4j) has to close it itself —
    and the shared SQLAlchemy pool second, since the record repositories borrow from it.
    Without this the pool was never disposed: connections stayed open until the process died,
    which a reloading dev server or a test run does repeatedly.
    """
    yield
    await shutdown_repositories()
    await dispose_engine()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# slowapi reads the limiter off app.state — both the `@limiter.limit(...)` decorators in the
# routers and SlowAPIMiddleware (which applies RATE_LIMIT_DEFAULT as the global safeguard).
app.state.limiter = limiter

# Middleware added last sits outermost. CORS goes outside everything so preflight is answered
# before rate limiting, and error responses still carry the Access-Control-* headers.
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

# AppException carries its own status code and error code; everything else becomes a generic
# 500 so no internal detail (connection strings, stack traces) reaches the client.
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)
app.include_router(summarize.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(qa.router, prefix=settings.API_V1_STR)
app.include_router(translations.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
