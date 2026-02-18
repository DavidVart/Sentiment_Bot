"""Kalshi websocket consumer for real-time updates (optional, feature-flagged)."""

from __future__ import annotations

from typing import Callable

from src.utils.logging_utils import get_logger
from src.utils.schemas import PMPrice

logger = get_logger(__name__)


def consume_kalshi_ws(
    tickers: list[str],
    on_price: Callable[[PMPrice], None],
    ws_url: str = "wss://trading-api.kalshi.com/trade-api/ws/v2",
) -> None:
    """
    Optional websocket consumer for Kalshi. Not required for Phase 1 backfill.
    """
    logger.info(
        "Kalshi WS stub: tickers=%s (use REST for Phase 1)",
        tickers[:3],
    )
    raise NotImplementedError(
        "Websocket consumer is optional for Phase 1; use REST backfill and incremental poll."
    )
