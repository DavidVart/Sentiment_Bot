"""Polymarket websocket consumer for real-time price updates (optional, feature-flagged)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.utils.logging_utils import get_logger
from src.utils.schemas import PMPrice

logger = get_logger(__name__)


def consume_polymarket_ws(
    token_ids: list[str],
    on_price: Callable[[PMPrice], None],
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
) -> None:
    """
    Optional websocket consumer: subscribe to market updates for token_ids
    and call on_price for each price update. Intended for paper/live loop.
    Not required for backfill; use REST + incremental poll instead.
    """
    # Stub: full implementation would use websockets library and map
    # message payloads to PMPrice then call on_price(price).
    logger.info(
        "Polymarket WS stub: token_ids=%s, ws_url=%s (use REST backfill for Phase 1)",
        token_ids[:3],
        ws_url,
    )
    raise NotImplementedError(
        "Websocket consumer is optional for Phase 1; use REST backfill and incremental poll."
    )
