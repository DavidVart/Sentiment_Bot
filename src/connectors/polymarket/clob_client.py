"""Polymarket CLOB client: live pricing and historical timeseries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from py_clob_client.client import ClobClient as PyClobClient

from src.utils.http_utils import with_retry
from src.utils.logging_utils import get_logger
from src.utils.schemas import PMPrice

logger = get_logger(__name__)

CLOB_DEFAULT_HOST = "https://clob.polymarket.com"


class ClobClient:
    """Read-only CLOB access: live price/mid/book + historical prices-history."""

    def __init__(self, host: str = CLOB_DEFAULT_HOST) -> None:
        self.host = host.rstrip("/")
        self._client = PyClobClient(self.host)

    def get_midpoint(self, token_id: str) -> float | None:
        """Mid market price for token. Returns None on error."""
        def _get() -> float | None:
            raw = self._client.get_midpoint(token_id)
            if isinstance(raw, dict) and "mid" in raw:
                return float(raw["mid"])
            if isinstance(raw, (int, float)):
                return float(raw)
            return None
        return with_retry(_get)

    def get_price(self, token_id: str, side: str = "BUY") -> float | None:
        """Market price for token and side (BUY/SELL)."""
        def _get() -> float | None:
            raw = self._client.get_price(token_id, side)
            if isinstance(raw, dict) and "price" in raw:
                return float(raw["price"])
            if isinstance(raw, (int, float)):
                return float(raw)
            return None
        return with_retry(_get)

    def get_order_book(self, token_id: str) -> dict[str, Any] | None:
        """Order book summary for token."""
        def _get() -> dict[str, Any] | None:
            ob = self._client.get_order_book(token_id)
            return ob if isinstance(ob, dict) else None
        return with_retry(_get)

    def get_prices_history(
        self,
        token_id: str,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        interval: str = "max",
        fidelity_minutes: int | None = None,
    ) -> list[PMPrice]:
        """Historical price series via GET /prices-history. Returns list of PMPrice.
        Preferred: use interval='max' (no startTs/endTs) to get full history.
        startTs/endTs can be used but the API rejects windows that are too long."""
        params: dict[str, Any] = {"market": token_id}
        if start_ts is not None or end_ts is not None:
            if start_ts is not None:
                params["startTs"] = int(start_ts.timestamp())
            if end_ts is not None:
                params["endTs"] = int(end_ts.timestamp())
        else:
            params["interval"] = interval
        if fidelity_minutes is not None:
            params["fidelity"] = fidelity_minutes

        def _request() -> list[dict]:
            url = f"{self.host}/prices-history"
            with httpx.Client(timeout=60.0) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "history" in data:
                return data["history"] if isinstance(data["history"], list) else []
            return []

        try:
            raw = _request()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                logger.debug("prices-history %s: %s", token_id[:24], e.response.status_code)
                return []
            raise
        out: list[PMPrice] = []
        for row in raw:
            if isinstance(row, dict):
                t = row.get("t") or row.get("timestamp")
                p = row.get("p") or row.get("price")
                if p is None:
                    continue
                if isinstance(t, int):
                    # API returns t in seconds (per Polymarket docs)
                    ts = datetime.fromtimestamp(t if t < 2e12 else t / 1000.0, tz=timezone.utc)
                elif isinstance(t, str):
                    from dateutil import parser as date_parser
                    ts = date_parser.isoparse(t.replace("Z", "+00:00"))
                else:
                    continue
                out.append(
                    PMPrice(
                        token_id=token_id,
                        platform="polymarket",
                        ts=ts,
                        price=float(p),
                        mid=float(p),
                        source="rest",
                    )
                )
        return out
