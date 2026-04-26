"""
Rate limiting middleware for the Janus Dependency Analyzer REST API.

Implements a sliding-window rate limiter backed by an in-memory store.
Each API key (or IP address when no key is present) is tracked independently.

Default limits (configurable via environment variables):
  JANUS_RATE_LIMIT_REQUESTS  – max requests per window  (default: 60)
  JANUS_RATE_LIMIT_WINDOW    – window size in seconds    (default: 60)
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Deque, Dict, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_REQUESTS: int = int(os.environ.get("JANUS_RATE_LIMIT_REQUESTS", "60"))
DEFAULT_WINDOW_SECONDS: int = int(os.environ.get("JANUS_RATE_LIMIT_WINDOW", "60"))

# Paths that are exempt from rate limiting (health checks, docs, etc.)
EXEMPT_PATHS: frozenset = frozenset({
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
})


# ---------------------------------------------------------------------------
# Sliding-window counter
# ---------------------------------------------------------------------------

class SlidingWindowCounter:
    """
    Thread-safe sliding-window rate limiter for a single client.

    Stores the timestamps of recent requests in a deque and evicts entries
    that fall outside the current window on every check.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = Lock()

    def is_allowed(self) -> Tuple[bool, int, int]:
        """
        Record a request attempt and decide whether it is allowed.

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Evict expired timestamps
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            current_count = len(self._timestamps)

            if current_count < self.max_requests:
                self._timestamps.append(now)
                remaining = self.max_requests - current_count - 1
                return True, remaining, 0
            else:
                # Earliest request in the window determines when the client
                # can retry.
                oldest = self._timestamps[0]
                retry_after = int(oldest + self.window_seconds - now) + 1
                return False, 0, retry_after


# ---------------------------------------------------------------------------
# Rate limiter store
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Manages per-client SlidingWindowCounter instances.

    Clients are identified by their API key when present, falling back to
    their IP address.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._counters: Dict[str, SlidingWindowCounter] = {}
        self._lock = Lock()

    def check(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Check and record a request for the given client.

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        with self._lock:
            if client_id not in self._counters:
                self._counters[client_id] = SlidingWindowCounter(
                    self.max_requests, self.window_seconds
                )
            counter = self._counters[client_id]

        return counter.is_allowed()

    def reset(self, client_id: str) -> None:
        """Remove the counter for a client (useful in tests)."""
        with self._lock:
            self._counters.pop(client_id, None)


# ---------------------------------------------------------------------------
# Module-level default limiter
# ---------------------------------------------------------------------------

_default_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Return the module-level rate limiter instance."""
    return _default_limiter


# ---------------------------------------------------------------------------
# Starlette middleware
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that enforces per-client rate limits.

    Adds the following response headers on every non-exempt request:
      X-RateLimit-Limit     – maximum requests per window
      X-RateLimit-Remaining – requests remaining in the current window
      X-RateLimit-Reset     – seconds until the window resets (on 429 only)
    """

    def __init__(
        self,
        app: ASGIApp,
        limiter: Optional[RateLimiter] = None,
    ) -> None:
        super().__init__(app)
        self.limiter = limiter or _default_limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        client_id = self._get_client_id(request)
        allowed, remaining, retry_after = self.limiter.check(client_id)

        if not allowed:
            logger.warning(
                "Rate limit exceeded for client %s on %s %s",
                client_id,
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": (
                        f"Too many requests. "
                        f"Retry after {retry_after} second(s)."
                    ),
                },
                headers={
                    "X-RateLimit-Limit": str(self.limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                    "Retry-After": str(retry_after),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _get_client_id(request: Request) -> str:
        """
        Derive a stable client identifier from the request.

        Prefers the API key (already validated by the auth dependency) so that
        multiple IPs sharing a key share the same quota.  Falls back to the
        client IP address.
        """
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Use a short hash to avoid storing the raw key in the counter map
            import hashlib
            return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Respect X-Forwarded-For when behind a proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return "ip:" + forwarded_for.split(",")[0].strip()

        client = request.client
        if client:
            return f"ip:{client.host}"

        return "ip:unknown"
