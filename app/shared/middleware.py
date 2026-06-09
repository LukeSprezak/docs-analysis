import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.shared.context import generate_request_id, set_request_id

logger = logging.getLogger("app.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", generate_request_id())

        set_request_id(request_id)

        start_time = time.time()

        logger.info(f"Started {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"Finished {request.method} {request.url.path} "
                f"status={response.status_code} duration={process_time:.3f}s"
            )

            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.exception(
                f"Failed {request.method} {request.url.path} "
                f"duration={process_time:.3f}s error={e!s}"
            )
            raise
