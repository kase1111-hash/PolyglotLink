"""
PolyglotLink Security Middleware

Provides API key authentication, rate limiting, and request size
enforcement for the FastAPI application.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from polyglotlink.utils.logging import get_logger

logger = get_logger(__name__)

# Paths that are always unauthenticated (health probes)
_PUBLIC_PATHS = frozenset({"/health", "/ready", "/api/v1/health"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates API key on non-public endpoints when authentication is enabled."""

    async def dispatch(self, request: Request, call_next):
        settings = getattr(request.app.state, "settings", None)
        if settings is None or not settings.security.api_key_required:
            return await call_next(request)

        # Allow health/ready probes without auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        expected_key = settings.security.api_key
        if not expected_key:
            # api_key_required is True but no key configured — reject everything
            logger.error("API key required but not configured")
            return JSONResponse(
                status_code=500,
                content={"error": "server_misconfigured"},
            )

        header_name = settings.security.api_key_header
        provided_key = request.headers.get(header_name)

        if not provided_key or provided_key != expected_key:
            logger.warning(
                "Authentication failed",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Invalid or missing API key"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP token-bucket rate limiter."""

    def __init__(self, app, max_per_minute: int = 1000):
        super().__init__(app)
        self.max_per_minute = max_per_minute
        # {ip: (remaining_tokens, last_refill_time)}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._last_cleanup = time.monotonic()

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting on health probes
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Periodic cleanup of stale buckets (every 5 minutes)
        if now - self._last_cleanup > 300:
            self._cleanup(now)

        # Get or create bucket
        if client_ip in self._buckets:
            tokens, last_refill = self._buckets[client_ip]
            # Refill tokens based on elapsed time
            elapsed = now - last_refill
            tokens = min(self.max_per_minute, tokens + elapsed * (self.max_per_minute / 60.0))
        else:
            tokens = float(self.max_per_minute)
            last_refill = now

        if tokens < 1.0:
            logger.warning("Rate limit exceeded", client=client_ip, path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "message": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )

        # Consume a token
        self._buckets[client_ip] = (tokens - 1.0, now)
        return await call_next(request)

    def _cleanup(self, now: float) -> None:
        """Remove buckets that haven't been seen in 5 minutes."""
        stale = [ip for ip, (_, t) in self._buckets.items() if now - t > 300]
        for ip in stale:
            del self._buckets[ip]
        self._last_cleanup = now


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests that exceed the configured size limit."""

    def __init__(self, app, max_size_bytes: int = 1_048_576):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "message": f"Request body exceeds {self.max_size_bytes} bytes",
                },
            )
        return await call_next(request)
