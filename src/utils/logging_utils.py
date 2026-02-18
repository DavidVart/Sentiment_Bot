"""Structured logging for the pipeline."""

from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a logger with a single handler and level."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        log.addHandler(h)
    log.setLevel(level)
    return log
