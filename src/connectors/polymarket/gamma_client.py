"""Polymarket Gamma API client for market/event discovery."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from dateutil import parser as date_parser

from src.utils.http_utils import with_retry
from src.utils.logging_utils import get_logger
from src.utils.schemas import PMEvent, PMMarket

logger = get_logger(__name__)

GAMMA_DEFAULT_BASE = "https://gamma-api.polymarket.com"


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return date_parser.isoparse(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _token_ids_from_market(m: dict[str, Any]) -> list[str]:
    """Extract CLOB token IDs from a Gamma market object."""
    raw = m.get("clobTokenIds") or m.get("clobTokenIds")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        # Can be JSON array string or comma-separated
        raw = raw.strip()
        if raw.startswith("["):
            import json
            try:
                out = json.loads(raw)
                return out if isinstance(out, list) else []
            except json.JSONDecodeError:
                pass
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _outcome_names_from_market(m: dict[str, Any]) -> list[str]:
    raw = m.get("outcomes") or m.get("shortOutcomes")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


class GammaClient:
    """Read-only client for Polymarket Gamma API (events + markets)."""

    def __init__(self, base_url: str = GAMMA_DEFAULT_BASE) -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        def _request() -> Any:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(url, params=params or {})
                r.raise_for_status()
                return r.json()
        return with_retry(_request)

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: bool | None = None,
        order: str = "startDate",
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch events with pagination."""
        params: dict[str, Any] = {"limit": limit, "offset": offset, "order": order, "ascending": str(ascending).lower()}
        if closed is not None:
            params["closed"] = str(closed).lower()
        return self._get("/events", params) or []

    def fetch_all_events(
        self,
        closed: bool | None = None,
        max_pages: int = 50,
        order: str = "startDate",
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        """Paginate through all events."""
        out: list[dict[str, Any]] = []
        offset = 0
        limit = 100
        for _ in range(max_pages):
            batch = self.get_events(limit=limit, offset=offset, closed=closed, order=order, ascending=ascending)
            if not batch:
                break
            out.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return out

    def normalize_event(self, e: dict[str, Any]) -> PMEvent:
        """Convert Gamma event to canonical PMEvent."""
        event_id = str(e.get("id") or "")
        return PMEvent(
            event_id=f"polymarket_{event_id}",
            platform="polymarket",
            title=str(e.get("title") or ""),
            category=str(e.get("category") or ""),
            start_ts=_parse_iso(e.get("startDate") or e.get("startTime")),
            end_ts=_parse_iso(e.get("endDate") or e.get("closedTime")),
            status="closed" if e.get("closed") else "open",
            resolution_ts=_parse_iso(e.get("closedTime") or e.get("endDate")),
        )

    def normalize_markets_for_event(self, event_id: str, e: dict[str, Any]) -> list[PMMarket]:
        """Convert Gamma event's markets to canonical PMMarket list."""
        markets = e.get("markets") or []
        result: list[PMMarket] = []
        for m in markets:
            mid = str(m.get("id") or "")
            token_ids = _token_ids_from_market(m)
            outcome_names = _outcome_names_from_market(m)
            vol = m.get("volumeNum") or m.get("volume")
            if isinstance(vol, str):
                try:
                    vol = float(vol)
                except ValueError:
                    vol = None
            liq = m.get("liquidityNum") or m.get("liquidity")
            if isinstance(liq, str):
                try:
                    liq = float(liq)
                except ValueError:
                    liq = None
            result.append(
                PMMarket(
                    market_id=f"polymarket_{mid}",
                    event_id=event_id,
                    platform="polymarket",
                    slug=str(m.get("slug") or ""),
                    outcome_names=outcome_names,
                    token_ids=token_ids,
                    active=bool(m.get("active", True)),
                    volume=float(vol) if vol is not None else None,
                    liquidity=float(liq) if liq is not None else None,
                )
            )
        return result
