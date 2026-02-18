"""Rate limiter for APIs (e.g. Polygon free tier: 5 calls/min)."""

from __future__ import annotations

import time
from threading import Lock


class RateLimiter:
    """Enforce a minimum interval between calls (e.g. 12s for 5 calls/min)."""

    def __init__(self, calls_per_minute: int = 5) -> None:
        self._interval = 60.0 / calls_per_minute if calls_per_minute else 0.0
        self._last_called: float = 0.0
        self._lock = Lock()

    def wait_if_needed(self) -> None:
        """Block until at least _interval seconds have passed since last call."""
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_called
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_called = time.monotonic()
