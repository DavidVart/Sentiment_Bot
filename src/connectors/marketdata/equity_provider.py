"""Equity daily OHLCV: Polygon.io (primary) and yfinance (fallback)."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from src.utils.http_utils import with_retry
from src.utils.logging_utils import get_logger
from src.utils.rate_limit import RateLimiter
from src.utils.schemas import EquityBar

logger = get_logger(__name__)

POLYGON_BASE = "https://api.polygon.io"
CALLS_PER_MINUTE = 5


def _load_universe() -> list[str]:
    """Load symbols from configs/universe.yaml."""
    # __file__ = .../src/connectors/marketdata/equity_provider.py -> 4 parents = repo root
    config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs"
    path = config_dir / "universe.yaml"
    if not path.exists():
        return ["SPY", "QQQ", "AAPL"]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("underlyings", ["SPY", "QQQ", "AAPL"]))


class EquityProviderProtocol(Protocol):
    """Protocol for equity bar providers."""

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[EquityBar]:
        """Fetch daily OHLCV bars for symbol in [start, end]. Returns normalized list."""
        ...


class PolygonEquityProvider:
    """Polygon.io/Massive REST API for daily aggregates. Rate-limited to 5 calls/min."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = POLYGON_BASE,
        calls_per_minute: int = CALLS_PER_MINUTE,
    ) -> None:
        self.api_key = api_key or os.environ.get("MASSIVE_API", "")
        self.base_url = base_url.rstrip("/")
        self._rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self._rate_limiter.wait_if_needed()
        url = f"{self.base_url}{path}"
        params = dict(params or {})
        if self.api_key:
            params["apiKey"] = self.api_key

        def _request() -> Any:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                return r.json()

        return with_retry(_request)

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[EquityBar]:
        """GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}. Returns list of EquityBar."""
        if not self.api_key:
            logger.warning("MASSIVE_API not set; Polygon provider will likely fail")
        path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        try:
            data = self._get(path)
        except Exception as e:
            logger.warning("Polygon fetch_daily_bars %s failed: %s", symbol, e)
            return []
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            return []
        out: list[EquityBar] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            t_ms = r.get("t")
            if t_ms is None:
                continue
            ts_dt = datetime.fromtimestamp(int(t_ms) / 1000.0, tz=timezone.utc)
            bar_date = ts_dt.date()
            o = r.get("o")
            h = r.get("h")
            l_ = r.get("l")
            c = r.get("c")
            v = r.get("v")
            vw = r.get("vw")
            if o is None or h is None or l_ is None or c is None or v is None:
                continue
            out.append(
                EquityBar(
                    symbol=symbol,
                    ts=bar_date,
                    open=float(o),
                    high=float(h),
                    low=float(l_),
                    close=float(c),
                    volume=float(v),
                    vwap=float(vw) if vw is not None else None,
                    source="polygon",
                )
            )
        logger.info("Polygon: fetched %s bars for %s", len(out), symbol)
        return out


class YFinanceEquityProvider:
    """yfinance fallback for daily OHLCV (no API key, no strict rate limit)."""

    def __init__(self) -> None:
        try:
            import yfinance as yf  # noqa: F401
        except ImportError:
            raise ImportError("yfinance is required for YFinanceEquityProvider; pip install yfinance")

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[EquityBar]:
        """Fetch daily bars via yfinance. Returns list of EquityBar."""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
        except Exception as e:
            logger.warning("yfinance fetch_daily_bars %s failed: %s", symbol, e)
            return []
        if df is None or df.empty:
            return []
        out: list[EquityBar] = []
        for dt, row in df.iterrows():
            if hasattr(dt, "date"):
                bar_date = dt.date()
            else:
                bar_date = datetime.fromisoformat(str(dt)).date()
            open_ = row.get("Open")
            high = row.get("High")
            low = row.get("Low")
            close = row.get("Close")
            volume = row.get("Volume")
            if open_ is None or high is None or low is None or close is None:
                continue
            volume_val = float(volume) if volume is not None else 0.0
            out.append(
                EquityBar(
                    symbol=symbol,
                    ts=bar_date,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=volume_val,
                    vwap=None,  # yfinance history does not always expose VWAP
                    source="yfinance",
                )
            )
        logger.info("yfinance: fetched %s bars for %s", len(out), symbol)
        return out


def get_equity_provider(
    prefer: str = "polygon",
    polygon_api_key: str | None = None,
    polygon_base_url: str = POLYGON_BASE,
) -> EquityProviderProtocol:
    """Return primary provider (polygon if key set, else yfinance)."""
    if prefer == "polygon" and (polygon_api_key or os.environ.get("MASSIVE_API")):
        return PolygonEquityProvider(
            api_key=polygon_api_key or os.environ.get("MASSIVE_API"),
            base_url=polygon_base_url,
        )
    return YFinanceEquityProvider()
