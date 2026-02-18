"""Build options_features from options_snapshots + equity_bars: ATM IV, term structure, skew, realized vol, VIX."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from src.db import get_connection
from src.ingestion.pm_writer import write_options_features
from src.utils.logging_utils import get_logger
from src.utils.schemas import OptionsFeature

logger = get_logger(__name__)

# Expiry buckets: days to expiry ranges for 7D / 14D / 30D
DAYS_7D = (3, 12)
DAYS_14D = (13, 21)
DAYS_30D = (22, 45)


def _annualized_realized_vol(log_returns: list[float]) -> float | None:
    """Annualized realized vol: std(log returns) * sqrt(252)."""
    if not log_returns or len(log_returns) < 2:
        return None
    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((x - mean) ** 2 for x in log_returns) / (n - 1) if n > 1 else 0.0
    if variance <= 0:
        return 0.0
    return float(math.sqrt(variance * 252))


def _get_vix_close(feature_date: date) -> float | None:
    """Fetch VIX closing price for date via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; VIX close skipped")
        return None
    try:
        ticker = yf.Ticker("^VIX")
        # request single day and adjacent for weekend/holiday
        start = feature_date - timedelta(days=5)
        end = feature_date + timedelta(days=1)
        hist = ticker.history(start=start, end=end, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        # find row for feature_date
        for dt, row in hist.iterrows():
            d = dt.date() if hasattr(dt, "date") else date.fromisoformat(str(dt)[:10])
            if d == feature_date:
                close = row.get("Close")
                return float(close) if close is not None else None
        return None
    except Exception as e:
        logger.debug("VIX fetch failed for %s: %s", feature_date, e)
        return None


def run_build_options_features(
    underlying: str | None = None,
    feature_date: date | None = None,
    schema_version: int = 1,
) -> None:
    """
    Compute options_features from options_snapshots and equity_bars.
    If underlying is None, process all underlyings with snapshot data.
    If feature_date is None, process all snapshot_date values.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if underlying and feature_date:
                cur.execute(
                    "SELECT 1 FROM options_snapshots WHERE underlying = %s AND snapshot_date = %s LIMIT 1",
                    (underlying, feature_date),
                )
                if not cur.fetchone():
                    logger.debug("No options_snapshots for %s on %s", underlying, feature_date)
                    return
                keys = [(underlying, feature_date)]
            elif underlying:
                cur.execute(
                    "SELECT DISTINCT underlying, snapshot_date FROM options_snapshots WHERE underlying = %s ORDER BY snapshot_date",
                    (underlying,),
                )
                keys = list(cur.fetchall())
            elif feature_date:
                cur.execute(
                    "SELECT DISTINCT underlying, snapshot_date FROM options_snapshots WHERE snapshot_date = %s ORDER BY underlying",
                    (feature_date,),
                )
                keys = list(cur.fetchall())
            else:
                cur.execute(
                    "SELECT DISTINCT underlying, snapshot_date FROM options_snapshots ORDER BY underlying, snapshot_date"
                )
                keys = list(cur.fetchall())

    for (sym, snap_date) in keys:
        _build_for_underlying_date(sym, snap_date, schema_version)


def _build_for_underlying_date(underlying: str, feature_date: date, schema_version: int) -> None:
    """Compute one OptionsFeature row for (underlying, feature_date)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT expiry, strike, option_type, iv, delta
                FROM options_snapshots
                WHERE underlying = %s AND snapshot_date = %s AND iv IS NOT NULL
                """,
                (underlying, feature_date),
            )
            options_rows = cur.fetchall()
            cur.execute(
                "SELECT close FROM equity_bars WHERE symbol = %s AND ts <= %s ORDER BY ts DESC LIMIT 1",
                (underlying, feature_date),
            )
            spot_row = cur.fetchone()
            cur.execute(
                """
                SELECT ts, close FROM equity_bars
                WHERE symbol = %s AND ts <= %s AND ts >= %s
                ORDER BY ts
                """,
                (underlying, feature_date, feature_date - timedelta(days=65)),
            )
            equity_rows = cur.fetchall()

    spot = float(spot_row[0]) if spot_row else None
    if spot is None and options_rows:
        logger.debug("No spot for %s on %s; skipping options features", underlying, feature_date)
        return
    # If no options data, we still build row with realized_vol and vix_close only

    # ATM IV per expiry bucket (7D, 14D, 30D)
    atm_iv_7d = _atm_iv_for_bucket(options_rows, feature_date, spot, DAYS_7D)
    atm_iv_14d = _atm_iv_for_bucket(options_rows, feature_date, spot, DAYS_14D)
    atm_iv_30d = _atm_iv_for_bucket(options_rows, feature_date, spot, DAYS_30D)

    iv_term_slope = None
    if atm_iv_30d is not None and atm_iv_7d is not None:
        iv_term_slope = atm_iv_30d - atm_iv_7d

    iv_skew = _compute_iv_skew(options_rows, feature_date, spot)

    # Realized vol 5/10/20/60 day windows
    realized_vol_5d = _realized_vol_for_window(equity_rows, feature_date, 5)
    realized_vol_10d = _realized_vol_for_window(equity_rows, feature_date, 10)
    realized_vol_20d = _realized_vol_for_window(equity_rows, feature_date, 20)
    realized_vol_60d = _realized_vol_for_window(equity_rows, feature_date, 60)

    vix_close = _get_vix_close(feature_date)

    row = OptionsFeature(
        underlying=underlying,
        feature_date=feature_date,
        atm_iv_7d=atm_iv_7d,
        atm_iv_14d=atm_iv_14d,
        atm_iv_30d=atm_iv_30d,
        iv_term_slope=iv_term_slope,
        iv_skew=iv_skew,
        realized_vol_5d=realized_vol_5d,
        realized_vol_10d=realized_vol_10d,
        realized_vol_20d=realized_vol_20d,
        realized_vol_60d=realized_vol_60d,
        vix_close=vix_close,
        schema_version=schema_version,
    )
    with get_connection() as conn:
        write_options_features(conn, [row], schema_version=schema_version)
    logger.info("Built options_features for %s on %s", underlying, feature_date)


def _days_to_expiry(expiry: date, snapshot_date: date) -> int:
    return (expiry - snapshot_date).days


def _atm_iv_for_bucket(
    options_rows: list[Any],
    snapshot_date: date,
    spot: float | None,
    day_range: tuple[int, int],
) -> float | None:
    """Return ATM IV for contracts with days_to_expiry in day_range; use strike closest to spot."""
    lo, hi = day_range
    candidates = []
    for r in options_rows:
        expiry, strike_raw, option_type, iv, delta = r[0], r[1], r[2], r[3], r[4]
        strike = float(strike_raw) if isinstance(strike_raw, (int, float)) else None
        if strike is None or iv is None:
            continue
        dte = _days_to_expiry(expiry, snapshot_date)
        if not (lo <= dte <= hi):
            continue
        # Prefer calls for ATM; same strike for put would give similar IV
        if spot is not None:
            candidates.append((abs(strike - spot), float(iv)))
        else:
            candidates.append((0.0, float(iv)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _compute_iv_skew(
    options_rows: list[Any],
    snapshot_date: date,
    spot: float | None,
) -> float | None:
    """IV skew = OTM put IV - OTM call IV. Use ~25-delta style: put with strike < spot, call with strike > spot."""
    if spot is None:
        return None
    # OTM put: put with strike < spot, take one closest to spot (e.g. max strike among puts below spot)
    put_ivs = []
    call_ivs = []
    for r in options_rows:
        expiry, strike_raw, option_type, iv, delta = r[0], r[1], r[2], r[3], r[4]
        strike = float(strike_raw) if isinstance(strike_raw, (int, float)) else None
        if strike is None or iv is None:
            continue
        if option_type == "put" and strike < spot:
            put_ivs.append((strike, float(iv)))
        elif option_type == "call" and strike > spot:
            call_ivs.append((strike, float(iv)))
    if not put_ivs or not call_ivs:
        return None
    # Put: choose strike closest to spot (max strike below spot)
    put_ivs.sort(key=lambda x: x[0], reverse=True)
    otm_put_iv = put_ivs[0][1]
    # Call: choose strike closest to spot (min strike above spot)
    call_ivs.sort(key=lambda x: x[0])
    otm_call_iv = call_ivs[0][1]
    return otm_put_iv - otm_call_iv


def _realized_vol_for_window(
    equity_rows: list[Any],
    feature_date: date,
    window_days: int,
) -> float | None:
    """Annualized realized vol over the last window_days trading days."""
    if not equity_rows or len(equity_rows) < 2:
        return None
    # equity_rows are (ts, close) ordered by ts ascending
    cutoff = feature_date - timedelta(days=window_days + 5)
    window = [(r[0], float(r[1])) for r in equity_rows if r[0] >= cutoff and r[0] <= feature_date]
    if len(window) < 2:
        return None
    # take last (window_days + 1) points so we have window_days returns
    window = window[-(window_days + 1) :]
    if len(window) < 2:
        return None
    log_returns = []
    for i in range(1, len(window)):
        prev_close = window[i - 1][1]
        curr_close = window[i][1]
        if prev_close <= 0:
            continue
        log_returns.append(math.log(curr_close / prev_close))
    return _annualized_realized_vol(log_returns)
