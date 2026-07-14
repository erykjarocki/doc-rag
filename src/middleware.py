"""FastAPI middleware for request tracing and logging.

Provides automatic request ID injection, timing, and structured logging
for every HTTP request flowing through the API.

Usage:
    from src.middleware import RequestTracingMiddleware
    app.add_middleware(RequestTracingMiddleware)
"""

from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.log import generate_request_id, get_logger, request_id_var

logger = get_logger("src.middleware")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Injects request IDs, logs timing, and captures errors.

    Every request gets a unique 12-char hex ID propagated via contextvar
    so all downstream log calls include it automatically. On completion,
    a structured log line records method, path, status, and duration.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Use existing X-Request-ID header if present, else generate one
        req_id = request.headers.get("x-request-id", generate_request_id())
        token = request_id_var.set(req_id)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception(
                "Unhandled error in %s %s",
                request.method,
                request.url.path,
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            request_id_var.reset(token)

            # Skip noisy health-check logging
            if request.url.path != "/health":
                log_fn = logger.warning if status_code >= 400 else logger.info
                log_fn(
                    "%s %s -> %d (%.1fms)",
                    request.method,
                    request.url.path,
                    status_code,
                    duration_ms,
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": round(duration_ms, 1),
                        "request_id": req_id,
                    },
                )

            # Propagate request ID to response header
            response.headers["X-Request-ID"] = req_id
