"""HTTP client with exponential backoff and rate-limit respect."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

import httpx

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_MAX_RETRIES = 5


# Client errors that should not be retried (auth / bad request)
_NO_RETRY_STATUS_CODES = (400, 401, 403, 404, 422)


def with_retry(
    fn: Callable[[], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    retry_on: tuple[type[Exception], ...] = (httpx.HTTPError, ConnectionError),
) -> T:
    """Call fn; on retry_on exceptions, exponential backoff and retry.
    Do not retry on 400/401/403/404 (client errors)."""
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except retry_on as e:
            last = e
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in _NO_RETRY_STATUS_CODES:
                raise
            if attempt == max_retries - 1:
                raise
            delay = backoff_base ** attempt
            # Respect Retry-After header on 429
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
            logger.warning("Attempt %s failed: %s; retrying in %.1fs", attempt + 1, e, delay)
            time.sleep(delay)
    raise last  # type: ignore
