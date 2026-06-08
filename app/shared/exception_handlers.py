import logging
from typing import cast

from fastapi import Request
from fastapi.responses import JSONResponse

from app.shared.context import get_request_id
from app.shared.exceptions import AppException

logger = logging.getLogger("app.exception_handler")


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    exc = cast(AppException, exc)
    logger.error(
        f"AppException: {exc.message} "
        f"status_code={exc.status_code} "
        f"error_code={exc.error_code} "
        f"context={exc.context}",
        extra={"error_code": exc.error_code, "context": exc.context},
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "error_code": exc.error_code,
                "request_id": get_request_id(),
                "context": exc.context,
            }
        },
    )


async def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    detail = getattr(exc, "detail", "rate limit exceeded")
    logger.warning(f"Rate limit exceeded: {detail}", extra={"request_id": get_request_id()})

    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Too many requests. Please slow down and try again later.",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "request_id": get_request_id(),
            }
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception: {exc!s}", extra={"request_id": get_request_id()})

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "error_code": "INTERNAL_SERVER_ERROR",
                "request_id": get_request_id(),
            }
        },
    )
