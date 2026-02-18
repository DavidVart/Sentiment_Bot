"""Incremental update: poll latest prices for configured token IDs and write to pm_prices."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.connectors.kalshi import KalshiClient
from src.connectors.polymarket import ClobClient
from src.db import get_connection
from src.ingestion.pm_writer import write_pm_prices
from src.utils.logging_utils import get_logger
from src.utils.schemas import PMPrice

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_mapping() -> list[dict[str, Any]]:
    """Load configs/mapping.yaml for token_ids to poll."""
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


def collect_polymarket_prices(clob_base: str, token_ids: list[str]) -> list[PMPrice]:
    """Current mid for each token_id from CLOB."""
    clob = ClobClient(host=clob_base)
    now = datetime.now(timezone.utc)
    out: list[PMPrice] = []
    for tid in token_ids:
        if not tid:
            continue
        try:
            mid = clob.get_midpoint(tid)
            if mid is not None:
                out.append(
                    PMPrice(token_id=tid, platform="polymarket", ts=now, price=mid, mid=mid, source="rest")
                )
        except Exception as err:
            logger.warning("Polymarket mid %s: %s", tid[:20], err)
    return out


def collect_kalshi_prices(base_url: str, tickers: list[str]) -> list[PMPrice]:
    """Current last price for each ticker from Kalshi (GET /markets/{ticker})."""
    client = KalshiClient(base_url=base_url)
    now = datetime.now(timezone.utc)
    out: list[PMPrice] = []
    for ticker in tickers:
        if not ticker:
            continue
        try:
            m = client.get_market(ticker)
            if m and ("last_price_dollars" in m or "last_price" in m):
                p = m.get("last_price_dollars") or m.get("last_price")
                if p is not None:
                    prob = float(p) if isinstance(p, (int, float)) else float(str(p))
                    out.append(
                        PMPrice(token_id=ticker, platform="kalshi", ts=now, price=prob, mid=prob, source="rest")
                    )
        except Exception as err:
            logger.warning("Kalshi market %s: %s", ticker, err)
    return out


def run_incremental(
    polymarket_token_ids: list[str] | None = None,
    kalshi_tickers: list[str] | None = None,
) -> None:
    """
    Poll current prices for given token IDs (and from mapping.yaml if None), write to pm_prices.
    """
    cfg = load_data_sources()
    clob_base = (cfg.get("polymarket") or {}).get("clob_base_url", "https://clob.polymarket.com")
    kalshi_base = (cfg.get("kalshi") or {}).get("base_url", "https://trading-api.kalshi.com")
    if polymarket_token_ids is None or kalshi_tickers is None:
        mapping = load_mapping()
        if polymarket_token_ids is None:
            polymarket_token_ids = []
            for entry in mapping:
                ids = (entry.get("token_ids") or {}).get("polymarket")
                if ids:
                    polymarket_token_ids.append(ids if isinstance(ids, str) else str(ids))
        if kalshi_tickers is None:
            kalshi_tickers = []
            for entry in mapping:
                ticker = (entry.get("token_ids") or {}).get("kalshi")
                if ticker:
                    kalshi_tickers.append(ticker if isinstance(ticker, str) else str(ticker))
    all_prices: list[PMPrice] = []
    if polymarket_token_ids:
        all_prices.extend(collect_polymarket_prices(clob_base, polymarket_token_ids))
    if kalshi_tickers:
        all_prices.extend(collect_kalshi_prices(kalshi_base, kalshi_tickers))
    if all_prices:
        with get_connection() as conn:
            write_pm_prices(conn, all_prices)
        logger.info("Incremental update: wrote %s price rows", len(all_prices))
    else:
        logger.info("Incremental update: no prices to write")
