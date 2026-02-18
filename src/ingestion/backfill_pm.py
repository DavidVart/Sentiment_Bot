"""Backfill prediction-market events, markets, and price history from Polymarket + Kalshi."""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from src.connectors.kalshi import KalshiClient
from src.connectors.polymarket import ClobClient, GammaClient
from src.db import apply_migrations, get_connection
from src.ingestion.pm_writer import write_pm_events, write_pm_markets, write_pm_prices
from src.utils.logging_utils import get_logger
from src.utils.schemas import PMEvent, PMMarket, PMPrice

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_mapping() -> list[dict[str, Any]]:
    """Load configs/mapping.yaml."""
    path = CONFIG_DIR / "mapping.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        return yaml.safe_load(f) or []


def load_data_sources() -> dict[str, Any]:
    """Load configs/data_sources.yaml."""
    path = CONFIG_DIR / "data_sources.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def backfill_polymarket(
    gamma_base: str,
    clob_base: str,
    closed: bool | None = True,
    max_events: int = 25,
    days_history: int = 365,
) -> None:
    """Backfill Polymarket: Gamma events/markets -> DB; CLOB prices-history -> DB."""
    gamma = GammaClient(base_url=gamma_base)
    clob = ClobClient(host=clob_base)
    # Fetch most recent closed events (descending) so we get markets with available CLOB history
    events_raw = gamma.fetch_all_events(closed=closed, order="startDate", ascending=False)
    events_raw = events_raw[:max_events]
    all_events: list[PMEvent] = []
    all_markets: list[PMMarket] = []
    token_ids_to_backfill: list[str] = []
    for e in events_raw:
        pe = gamma.normalize_event(e)
        all_events.append(pe)
        markets = gamma.normalize_markets_for_event(pe.event_id, e)
        all_markets.extend(markets)
        for m in markets:
            token_ids_to_backfill.extend(m.token_ids)
    token_ids_to_backfill = list(dict.fromkeys(token_ids_to_backfill))
    with get_connection() as conn:
        write_pm_events(conn, all_events)
        write_pm_markets(conn, all_markets)
    cutoff_ts = datetime.now(timezone.utc) - timedelta(days=days_history)
    all_prices: list[PMPrice] = []
    for tid in token_ids_to_backfill:
        if not tid:
            continue
        try:
            prices = clob.get_prices_history(tid, interval="max")
            # Filter to only keep prices within the requested window
            prices = [p for p in prices if p.ts >= cutoff_ts]
            all_prices.extend(prices)
        except Exception as err:
            logger.warning("get_prices_history %s: %s", tid[:20], err)
    if all_prices:
        with get_connection() as conn:
            write_pm_prices(conn, all_prices)
    logger.info("Polymarket backfill: %s events, %s markets, %s price rows", len(all_events), len(all_markets), len(all_prices))


def backfill_kalshi(
    base_url: str,
    status: str | None = "closed",
    max_events: int = 25,
    days_history: int = 365,
) -> None:
    """Backfill Kalshi: events/markets -> DB; candlesticks -> pm_prices."""
    client = KalshiClient(base_url=base_url)
    events_raw = client.fetch_all_events(status=status)
    events_raw = events_raw[:max_events]
    all_events: list[PMEvent] = []
    all_markets: list[PMMarket] = []
    # Track (ticker, series_ticker) pairs for candlestick fetching
    tickers_with_series: list[tuple[str, str]] = []
    for e in events_raw:
        pe = client.normalize_event(e)
        all_events.append(pe)
        markets_raw = e.get("markets") or []
        markets = client.normalize_markets_for_event(pe.event_id, e)
        all_markets.extend(markets)
        for m, m_raw in zip(markets, markets_raw):
            series = m_raw.get("series_ticker") or ""
            for tid in m.token_ids:
                tickers_with_series.append((tid, series))
    # Deduplicate by ticker
    seen: set[str] = set()
    unique_tickers: list[tuple[str, str]] = []
    for ticker, series in tickers_with_series:
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique_tickers.append((ticker, series))
    with get_connection() as conn:
        write_pm_events(conn, all_events)
        write_pm_markets(conn, all_markets)
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=days_history)
    # Kalshi's unauthenticated API is heavily rate-limited; cap candlestick fetches
    max_candle_markets = min(len(unique_tickers), 100)
    if len(unique_tickers) > max_candle_markets:
        logger.info("Kalshi: fetching candlesticks for %d of %d markets (rate limit)", max_candle_markets, len(unique_tickers))
    all_prices: list[PMPrice] = []
    for i, (ticker, series) in enumerate(unique_tickers[:max_candle_markets]):
        try:
            prices = client.get_market_candlesticks(
                ticker, start_ts=start_ts, end_ts=end_ts,
                period_interval_min=1440, series_ticker=series or None,
            )
            all_prices.extend(prices)
        except Exception as err:
            logger.warning("get_market_candlesticks %s: %s", ticker, err)
        # Kalshi public API rate-limits aggressively; throttle requests
        _time.sleep(0.5)
    if all_prices:
        with get_connection() as conn:
            write_pm_prices(conn, all_prices)
    logger.info("Kalshi backfill: %s events, %s markets, %s price rows", len(all_events), len(all_markets), len(all_prices))


def run_backfill(
    polymarket: bool = True,
    kalshi: bool = True,
    max_events_per_platform: int = 25,
    days_history: int = 365,
) -> None:
    """Run full backfill: migrations, then Polymarket and/or Kalshi."""
    apply_migrations()
    cfg = load_data_sources()
    gamma_base = (cfg.get("polymarket") or {}).get("gamma_base_url", "https://gamma-api.polymarket.com")
    clob_base = (cfg.get("polymarket") or {}).get("clob_base_url", "https://clob.polymarket.com")
    kalshi_base = (cfg.get("kalshi") or {}).get("base_url", "https://trading-api.kalshi.com")
    if polymarket:
        backfill_polymarket(gamma_base, clob_base, closed=True, max_events=max_events_per_platform, days_history=days_history)
    if kalshi:
        try:
            backfill_kalshi(kalshi_base, status="closed", max_events=max_events_per_platform, days_history=days_history)
        except Exception as e:
            status = getattr(e, "response", None) and getattr(e.response, "status_code", None)
            if status in (401, 403):
                logger.warning("Kalshi skipped (auth required: %s). Set KALSHI_API_KEY to backfill Kalshi.", e)
            else:
                raise
