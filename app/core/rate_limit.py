"""In-memory sliding-window rate limiter for authentication endpoints."""

from collections import defaultdict, deque
import threading
import time
from typing import Callable, Deque, Dict, Tuple

from fastapi import HTTPException, Request, status

from app.core.config import settings


class AuthRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by bucket and client IP."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, *, bucket: str, client_key: str, max_attempts: int, window_seconds: int) -> None:
        """Record an attempt and raise HTTP 429 if the window threshold is exceeded."""
        now = time.monotonic()
        cutoff = now - window_seconds
        key = (bucket, client_key)

        with self._lock:
            dq = self._requests[key]
            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= max_attempts:
                retry_after = max(1, int(dq[0] + window_seconds - now) + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many authentication attempts. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            dq.append(now)

    def reset(self) -> None:
        """Clear all recorded rate-limit counters (used in test isolation)."""
        with self._lock:
            self._requests.clear()


_auth_rate_limiter = AuthRateLimiter()


def get_auth_rate_limiter() -> AuthRateLimiter:
    """Return the singleton AuthRateLimiter instance."""
    return _auth_rate_limiter


def reset_auth_rate_limiter() -> None:
    """Reset the singleton auth rate limiter."""
    _auth_rate_limiter.reset()


def _extract_client_ip(request: Request) -> str:
    """Extract client IP address safely without trusting arbitrary spoofed headers in dev."""
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_auth(bucket: str) -> Callable[[Request], None]:
    """FastAPI dependency factory enforcing sliding-window rate limiting per auth bucket."""

    def _dependency(request: Request) -> None:
        client_ip = _extract_client_ip(request)
        _auth_rate_limiter.check(
            bucket=bucket,
            client_key=client_ip,
            max_attempts=settings.AUTH_RATE_LIMIT_MAX_ATTEMPTS,
            window_seconds=settings.AUTH_RATE_LIMIT_WINDOW_SECONDS,
        )

    return _dependency

