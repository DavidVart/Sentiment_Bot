"""Kalshi API client: events, markets, and historical candlesticks.

Authentication uses RSA-PSS signatures (SHA-256).  Required env vars:
    KALSHI_API_KEY            – API key ID (UUID)
    KALSHI_PRIVATE_KEY_PATH   – path to PEM private key file
"""

from __future__ import annotations

import base64
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dateutil import parser as date_parser

from src.utils.http_utils import with_retry
from src.utils.logging_utils import get_logger
from src.utils.schemas import PMEvent, PMMarket, PMPrice

logger = get_logger(__name__)

KALSHI_DEFAULT_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _load_private_key(key_path: str):
    """Load an RSA private key from a PEM file."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend(),
        )


def _sign_request(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """Create RSA-PSS signature for a Kalshi API request.

    The signed message is ``{timestamp}{METHOD}{path}`` where *path*
    is the full URI path (e.g. ``/trade-api/v2/events``) **without**
    query parameters.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    message = f"{timestamp_ms}{method}{path}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _parse_ts(ts: str | int | None) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, int):
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        return date_parser.isoparse(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _dollars_to_prob(raw: Any) -> float | None:
    """Kalshi returns prices in dollars (0–1 for binary)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return None
    return None


class KalshiClient:
    """Read-only client for Kalshi API (events, markets, candlesticks).

    Authenticates with RSA-PSS signed headers when ``KALSHI_API_KEY``
    and ``KALSHI_PRIVATE_KEY_PATH`` are set.  Falls back to unauthenticated
    requests (public endpoints only) when keys are absent.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or KALSHI_DEFAULT_BASE).rstrip("/")
        self._api_key = os.environ.get("KALSHI_API_KEY", "")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        self._private_key = None
        if self._api_key and key_path and Path(key_path).exists():
            try:
                self._private_key = _load_private_key(key_path)
                logger.info("Kalshi RSA auth enabled (key_id=%s…)", self._api_key[:8])
            except Exception as exc:
                logger.warning("Failed to load Kalshi private key: %s", exc)

    def _auth_headers(self, method: str, full_url: str) -> dict[str, str]:
        """Build authentication headers for a request."""
        h: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
        if not self._private_key or not self._api_key:
            return h

        timestamp_ms = int(time.time() * 1000)
        # Sign only the path portion (no query string)
        path = urlparse(full_url).path
        signature = _sign_request(self._private_key, timestamp_ms, method, path)

        h["KALSHI-ACCESS-KEY"] = self._api_key
        h["KALSHI-ACCESS-TIMESTAMP"] = str(timestamp_ms)
        h["KALSHI-ACCESS-SIGNATURE"] = signature
        return h

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"

        def _request() -> Any:
            headers = self._auth_headers("GET", url)
            with httpx.Client(timeout=30.0) as client:
                r = client.get(url, params=params or {}, headers=headers)
                r.raise_for_status()
                return r.json()

        return with_retry(_request)

    def get_events(
        self,
        limit: int = 200,
        cursor: str | None = None,
        with_nested_markets: bool = True,
        status: str | None = None,
    ) -> dict[str, Any]:
        """GET /events. Returns { events, cursor }."""
        params: dict[str, Any] = {"limit": limit, "with_nested_markets": with_nested_markets}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return self._get("/events", params) or {"events": [], "cursor": ""}

    def get_markets(self, event_ticker: str) -> list[dict[str, Any]]:
        """GET /events/{event_ticker}/markets."""
        raw = self._get(f"/events/{event_ticker}/markets")
        return raw.get("markets", raw) if isinstance(raw, dict) else (raw or [])

    def get_market(self, market_ticker: str) -> dict[str, Any] | None:
        """GET /markets/{market_ticker} for single market (e.g. current price)."""
        raw = self._get(f"/markets/{market_ticker}")
        return raw if isinstance(raw, dict) else None

    def get_market_candlesticks(
        self,
        ticker: str,
        start_ts: datetime,
        end_ts: datetime,
        period_interval_min: int = 1440,
        series_ticker: str | None = None,
    ) -> list[PMPrice]:
        """GET /series/{series}/markets/{ticker}/candlesticks. Returns PMPrice list.
        If series_ticker is None, infer from market ticker (strip last segment)."""
        if not series_ticker:
            series_ticker = self._infer_series_ticker(ticker)
        params = {
            "start_ts": int(start_ts.timestamp()),
            "end_ts": int(end_ts.timestamp()),
            "period_interval": period_interval_min,
        }
        raw = self._get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params)
        candles = raw.get("candlesticks", raw) if isinstance(raw, dict) else (raw or [])
        out: list[PMPrice] = []
        for c in candles:
            if not isinstance(c, dict):
                continue
            end_ts_val = c.get("end_period_ts") or c.get("end_ts") or c.get("endTs")
            # Extract yes price from nested price dict or top-level fields
            price_dict = c.get("price", {})
            if isinstance(price_dict, dict):
                yes_price = price_dict.get("close") or price_dict.get("mean") or price_dict.get("previous")
            else:
                yes_price = c.get("yes_price") or c.get("close") or c.get("last_price")
            p = _dollars_to_prob(yes_price)
            if p is None:
                continue
            # Kalshi prices are in cents (0-100), normalize to probability (0-1)
            if p > 1.0:
                p = p / 100.0
            ts = _parse_ts(end_ts_val)
            if ts is None:
                continue
            out.append(
                PMPrice(
                    token_id=ticker,
                    platform="kalshi",
                    ts=ts,
                    price=p,
                    mid=p,
                    source="rest",
                )
            )
        return out

    @staticmethod
    def _infer_series_ticker(ticker: str) -> str:
        """Infer series ticker from market ticker: e.g. KXIPOOLIPOP-25DEC01 -> KXIPOOLIPOP."""
        parts = ticker.split("-")
        if len(parts) >= 2:
            return parts[0]
        return ticker

    def fetch_all_events(self, status: str | None = None, max_pages: int = 100) -> list[dict[str, Any]]:
        """Paginate through all events."""
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            resp = self.get_events(cursor=cursor or None, status=status)
            events = resp.get("events") or []
            out.extend(events)
            cursor = resp.get("cursor") or ""
            if not cursor or not events:
                break
        return out

    def normalize_event(self, e: dict[str, Any]) -> PMEvent:
        """Convert Kalshi event to canonical PMEvent."""
        ticker = str(e.get("event_ticker") or "")
        return PMEvent(
            event_id=f"kalshi_{ticker}",
            platform="kalshi",
            title=str(e.get("title") or e.get("sub_title") or ""),
            category=str(e.get("category") or ""),
            start_ts=None,
            end_ts=_parse_ts(e.get("close_time") or e.get("expected_expiration_time")),
            status=str(e.get("status", "")).lower() or "open",
            resolution_ts=None,
        )

    def normalize_markets_for_event(self, event_id: str, e: dict[str, Any]) -> list[PMMarket]:
        """Convert Kalshi event's markets to canonical PMMarket list."""
        markets = e.get("markets") or []
        result: list[PMMarket] = []
        for m in markets:
            ticker = str(m.get("ticker") or "")
            # Token ID for Kalshi is the market ticker (yes outcome = probability).
            last = _dollars_to_prob(m.get("last_price_dollars") or m.get("last_price"))
            vol = m.get("volume") or m.get("volume_fp")
            if isinstance(vol, str):
                try:
                    vol = float(vol)
                except ValueError:
                    vol = None
            liq = _dollars_to_prob(m.get("liquidity_dollars") or m.get("liquidity"))
            if liq is not None:
                liq = float(liq)
            result.append(
                PMMarket(
                    market_id=f"kalshi_{ticker}",
                    event_id=event_id,
                    platform="kalshi",
                    slug=ticker,
                    outcome_names=["yes", "no"],
                    token_ids=[ticker],
                    active=(m.get("status") or "").lower() == "active",
                    volume=float(vol) if vol is not None else None,
                    liquidity=liq,
                )
            )
        return result
