"""Backfill equity daily OHLCV: Polygon first, yfinance fallback. Uses configs/universe.yaml."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.connectors.marketdata import PolygonEquityProvider, YFinanceEquityProvider
from src.db import apply_migrations, get_connection
from src.ingestion.pm_writer import write_equity_bars
from src.utils.logging_utils import get_logger
from src.utils.schemas import EquityBar

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_data_sources() -> dict[str, Any]:
    """Load configs/data_sources.yaml."""
    path = CONFIG_DIR / "data_sources.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_universe() -> list[str]:
    """Load symbols from configs/universe.yaml."""
    path = CONFIG_DIR / "universe.yaml"
    if not path.exists():
        return ["SPY", "QQQ", "AAPL"]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("underlyings", ["SPY", "QQQ", "AAPL"]))


def run_backfill_equity(
    years: int = 2,
    symbols: list[str] | None = None,
) -> None:
    """
    Backfill daily equity bars for 2 years (or years param).
    Tries Polygon first (rate-limited 5/min), falls back to yfinance per symbol.
    """
    apply_migrations()
    symbols = symbols or load_universe()
    end = date.today()
    start = end - timedelta(days=years * 365)
    cfg = load_data_sources()
    polygon_cfg = cfg.get("polygon") or {}
    base_url = polygon_cfg.get("base_url", "https://api.polygon.io")
    polygon = PolygonEquityProvider(base_url=base_url)
    yfinance_provider = YFinanceEquityProvider()
    all_bars: list[EquityBar] = []
    for symbol in symbols:
        bars = polygon.fetch_daily_bars(symbol, start, end)
        if not bars:
            logger.info("Polygon returned no bars for %s; using yfinance fallback", symbol)
            bars = yfinance_provider.fetch_daily_bars(symbol, start, end)
        all_bars.extend(bars)
    if all_bars:
        with get_connection() as conn:
            write_equity_bars(conn, all_bars)
        logger.info("Equity backfill complete: %s bars for %s symbols", len(all_bars), len(symbols))
    else:
        logger.warning("Equity backfill: no bars fetched for symbols %s", symbols)
