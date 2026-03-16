"""
Request/response logging middleware for the FastAPI app.

Logs every incoming request with timing, status code, and some
basic request metadata. Also handles CORS headers.
"""

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all HTTP requests with timing information.
    Assigns a request ID to each request for tracing.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.monotonic()

        # stash the request ID so downstream handlers can access it
        request.state.request_id = request_id

        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            request_id=request_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error(
                "http.error",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                error=str(exc),
                latency_ms=round(elapsed, 2),
            )
            raise

        elapsed = (time.monotonic() - start_time) * 1000

        logger.info(
            "http.response",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            request_id=request_id,
            latency_ms=round(elapsed, 2),
        )

        # inject request ID header for client-side correlation
        response.headers["X-Request-ID"] = request_id
        return response
