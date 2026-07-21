"""Shared request limiter (slowapi).

The limit key is the client IP address (`get_remote_address`). The global default limit acts
as a safety net for every endpoint, while tuned limits on the expensive ones (LLM, upload,
auth) are applied with the `@limiter.limit(...)` decorator in the routers.

The limiter is a single shared object — endpoints import this exact `limiter`, and tests
disable it (`limiter.enabled = False`) so repeated calls from the same address do not run
into the limit.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.shared.config import settings

# headers_enabled is deliberately False: injecting X-RateLimit-* headers would require a
# `response: Response` parameter on every protected endpoint. The limit works without them,
# and exceeding it returns a consistent 429 error (rate_limit_exceeded_handler).
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    enabled=settings.RATE_LIMIT_ENABLED,
)
