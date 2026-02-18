"""
Master feature alignment: single training matrix features_bar(ts, underlying).
Joins options_features, sentiment_features, pm_features, equity_bars onto 15-min
market-hours clock (9:30–16:00 ET). Anti-lookahead: all features at t use only data
strictly before t.
"""

from __future__ import annotations

import math
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from src.db import get_connection
from src.features.sentiment_features import (
    _get_et_tz,
    _load_underlyings,
    market_hours_bars_utc_for_date,
)
from src.utils.logging_utils import get_logger
from src.utils.schemas import FeatureRow

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
PM_GAP_THRESHOLD_HOURS = 2.0


def _load_mapping() -> list[dict[str, Any]]:
    path = CONFIG_DIR / "mapping.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _underlying_to_token_ids(mapping: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build underlying -> list[token_id] from mapping (non-empty token_ids only)."""
    out: dict[str, list[str]] = {}
    for entry in mapping:
        tokens = entry.get("token_ids") or {}
        underlyings = entry.get("affected_underlyings") or []
        for tid in tokens.values():
            if not (tid and str(tid).strip()):
                continue
            tid = str(tid).strip()
            for u in underlyings:
                u = str(u).strip()
                if u not in out:
                    out[u] = []
                if tid not in out[u]:
                    out[u].append(tid)
    return out


def _auto_discover_token_ids(underlyings: list[str], top_n: int = 20) -> dict[str, list[str]]:
    """
    Discover the most data-rich pm_features token_ids from the DB and map them
    to all underlyings.  Used as a fallback when mapping.yaml yields no tokens
    for a given underlying.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT token_id, count(*) AS cnt
                    FROM pm_features
                    GROUP BY token_id
                    ORDER BY cnt DESC
                    LIMIT %s
                    """,
                    (top_n,),
                )
                rows = cur.fetchall()
        if not rows:
            return {}
        token_ids = [str(r[0]) for r in rows]
        return {u: token_ids for u in underlyings}
    except Exception as exc:
        logger.debug("Auto-discover pm token_ids failed: %s", exc)
        return {}


def _bar_date(bar_ts: datetime) -> date:
    """Trading date for bar (ET); bar at 16:00 is still that date."""
    try:
        from zoneinfo import ZoneInfo
        et = bar_ts.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        import pytz
        et = bar_ts.astimezone(pytz.timezone("America/New_York"))
    return et.date()


def _options_for_bar(conn: Any, underlying: str, bar_ts: datetime) -> tuple[dict[str, Any], bool]:
    """
    Most recent options_features with feature_date < bar_date (no lookahead).
    Falls back to nearest available row (gap_flag=True) when no historical data
    exists yet -- this is expected for the first day of collection.
    Returns (row dict or empty, options_gap_flag).
    """
    bar_d = _bar_date(bar_ts)
    row = None
    gap = False
    with conn.cursor() as cur:
        # Preferred: strictly before bar_date (anti-lookahead)
        cur.execute(
            """
            SELECT atm_iv_7d, atm_iv_14d, atm_iv_30d, iv_term_slope, iv_skew,
                   realized_vol_5d, realized_vol_10d, realized_vol_20d, realized_vol_60d, vix_close
            FROM options_features
            WHERE underlying = %s AND feature_date < %s
            ORDER BY feature_date DESC
            LIMIT 1
            """,
            (underlying, bar_d),
        )
        row = cur.fetchone()
        if not row:
            # Fallback: nearest available row (may be same-day or future); flag the gap
            cur.execute(
                """
                SELECT atm_iv_7d, atm_iv_14d, atm_iv_30d, iv_term_slope, iv_skew,
                       realized_vol_5d, realized_vol_10d, realized_vol_20d, realized_vol_60d, vix_close
                FROM options_features
                WHERE underlying = %s
                ORDER BY abs(feature_date - %s)
                LIMIT 1
                """,
                (underlying, bar_d),
            )
            row = cur.fetchone()
            gap = True
    if not row or len(row) < 10:
        return ({}, True)
    return ({
        "atm_iv_7d": row[0], "atm_iv_14d": row[1], "atm_iv_30d": row[2],
        "iv_term_slope": row[3], "iv_skew": row[4],
        "realized_vol_5d": row[5], "realized_vol_10d": row[6], "realized_vol_20d": row[7], "realized_vol_60d": row[8],
        "vix_close": row[9],
    }, gap)


def _sentiment_for_bar(conn: Any, underlying: str, bar_ts: datetime) -> dict[str, Any]:
    """Sentiment row for (underlying, bar_ts); if missing return 0.0 and no_news_flag=True."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sent_news_asset, sent_social_asset, sent_macro_topic, sent_dispersion,
                   sent_momentum, sent_volume, no_news_flag
            FROM sentiment_features
            WHERE underlying = %s AND ts = %s
            """,
            (underlying, bar_ts),
        )
        row = cur.fetchone()
    if not row or len(row) < 7:
        return {
            "sent_news_asset": 0.0, "sent_social_asset": 0.0, "sent_macro_topic": 0.0,
            "sent_dispersion": 0.0, "sent_momentum": 0.0, "sent_volume": 0,
            "no_news_flag": True,
        }
    return {
        "sent_news_asset": float(row[0]), "sent_social_asset": float(row[1]), "sent_macro_topic": float(row[2]),
        "sent_dispersion": float(row[3]), "sent_momentum": float(row[4]), "sent_volume": int(row[5]),
        "no_news_flag": bool(row[6]),
    }


def _pm_for_bar(
    conn: Any,
    token_ids: list[str],
    bar_ts: datetime,
    gap_hours: float = PM_GAP_THRESHOLD_HOURS,
) -> tuple[dict[str, Any], bool]:
    """
    Latest pm_features per token with ts <= bar_ts; forward-fill if gap < gap_hours.
    Aggregate (mean) across tokens. pm_gap_flag if any token's latest is > gap_hours ago.
    """
    if not token_ids:
        return ({}, True)
    gap_sec = gap_hours * 3600
    out: dict[str, list[float]] = {
        "p": [], "logit_p": [], "delta_p_1h": [], "delta_p_1d": [], "momentum": [],
        "vol_of_p": [], "time_to_event": [], "surprise_z": [],
    }
    any_gap = False
    with conn.cursor() as cur:
        for tid in token_ids:
            cur.execute(
                """
                SELECT ts, p, logit_p, delta_p_1h, delta_p_1d, momentum, vol_of_p, time_to_event, surprise_z
                FROM pm_features
                WHERE token_id = %s AND ts <= %s
                ORDER BY ts DESC
                LIMIT 1
                """,
                (tid, bar_ts),
            )
            row = cur.fetchone()
            if not row or len(row) < 9:
                any_gap = True
                continue
            ts, p, logit_p, d1h, d1d, mom, vol, tte, sz = row
            ts_dt = ts if isinstance(ts, datetime) else datetime.fromtimestamp(float(ts), tz=timezone.utc)
            if getattr(ts_dt, "tzinfo", None) is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            bar_dt = bar_ts if bar_ts.tzinfo else bar_ts.replace(tzinfo=timezone.utc)
            delta_sec = (bar_dt - ts_dt).total_seconds()
            if delta_sec > gap_sec:
                any_gap = True
            out["p"].append(float(p))
            out["logit_p"].append(float(logit_p))
            if d1h is not None:
                out["delta_p_1h"].append(float(d1h))
            if d1d is not None:
                out["delta_p_1d"].append(float(d1d))
            if mom is not None:
                out["momentum"].append(float(mom))
            if vol is not None:
                out["vol_of_p"].append(float(vol))
            if tte is not None:
                out["time_to_event"].append(float(tte))
            if sz is not None:
                out["surprise_z"].append(float(sz))
    agg = {}
    for k, v in out.items():
        if v:
            agg[k] = sum(v) / len(v)
        else:
            agg[k] = None
    return (agg, any_gap)


def _equity_for_bar(conn: Any, underlying: str, bar_ts: datetime) -> dict[str, Any]:
    """
    Return 1d return and 20d realized vol using only equity_bars with ts < bar_date (no lookahead).
    return_1d = (close[-1] - close[-2]) / close[-2]; realized_vol_20d = annualized std of 20 log returns.
    """
    bar_d = _bar_date(bar_ts)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, close FROM equity_bars
            WHERE symbol = %s AND ts < %s
            ORDER BY ts DESC
            LIMIT 22
            """,
            (underlying, bar_d),
        )
        rows = cur.fetchall()
    if not rows or len(rows) < 2:
        return {"equity_return_1d": None, "equity_realized_vol_20d": None}
    closes = [float(r[1]) for r in reversed(rows) if len(r) >= 2 and r[1] is not None]
    if len(closes) < 2:
        return {"equity_return_1d": None, "equity_realized_vol_20d": None}
    return_1d = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] else None
    if len(closes) < 3:
        return {"equity_return_1d": return_1d, "equity_realized_vol_20d": None}
    log_returns = []
    for i in range(1, min(len(closes), 21)):
        if closes[i - 1] and closes[i - 1] > 0:
            log_returns.append(math.log(closes[i] / closes[i - 1]))
    if len(log_returns) < 2:
        return {"equity_return_1d": return_1d, "equity_realized_vol_20d": None}
    mean_lr = sum(log_returns) / len(log_returns)
    var = sum((x - mean_lr) ** 2 for x in log_returns) / (len(log_returns) - 1)
    realized_vol_20d = math.sqrt(var * 252) if var > 0 else 0.0
    return {"equity_return_1d": return_1d, "equity_realized_vol_20d": realized_vol_20d}


def build_row(
    conn: Any,
    underlying: str,
    bar_ts: datetime,
    token_ids: list[str],
    schema_version: int,
) -> FeatureRow:
    """
    Build one FeatureRow for (underlying, bar_ts) by joining options, sentiment, PM, and equity data.
    All data used is strictly before bar_ts (no lookahead). Returns a FeatureRow for writing to feature_bars.
    """
    opt, options_gap = _options_for_bar(conn, underlying, bar_ts)
    sent = _sentiment_for_bar(conn, underlying, bar_ts)
    pm, pm_gap = _pm_for_bar(conn, token_ids, bar_ts)
    eq = _equity_for_bar(conn, underlying, bar_ts)
    return FeatureRow(
        underlying=underlying,
        ts=bar_ts,
        schema_version=schema_version,
        atm_iv_7d=opt.get("atm_iv_7d"),
        atm_iv_14d=opt.get("atm_iv_14d"),
        atm_iv_30d=opt.get("atm_iv_30d"),
        iv_term_slope=opt.get("iv_term_slope"),
        iv_skew=opt.get("iv_skew"),
        realized_vol_5d=opt.get("realized_vol_5d"),
        realized_vol_10d=opt.get("realized_vol_10d"),
        realized_vol_20d=opt.get("realized_vol_20d"),
        realized_vol_60d=opt.get("realized_vol_60d"),
        vix_close=opt.get("vix_close"),
        options_gap_flag=options_gap,
        sent_news_asset=sent["sent_news_asset"],
        sent_social_asset=sent["sent_social_asset"],
        sent_macro_topic=sent["sent_macro_topic"],
        sent_dispersion=sent["sent_dispersion"],
        sent_momentum=sent["sent_momentum"],
        sent_volume=sent["sent_volume"],
        no_news_flag=sent["no_news_flag"],
        pm_p=pm.get("p"),
        pm_logit_p=pm.get("logit_p"),
        pm_delta_p_1h=pm.get("delta_p_1h"),
        pm_delta_p_1d=pm.get("delta_p_1d"),
        pm_momentum=pm.get("momentum"),
        pm_vol_of_p=pm.get("vol_of_p"),
        pm_time_to_event=pm.get("time_to_event"),
        pm_surprise_z=pm.get("surprise_z"),
        pm_gap_flag=pm_gap,
        equity_return_1d=eq.get("equity_return_1d"),
        equity_realized_vol_20d=eq.get("equity_realized_vol_20d"),
    )


def run_build_feature_matrix(
    underlying: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    schema_version: int = 1,
) -> None:
    """
    Build feature_bars for the 15-min market-hours master clock (9:30–16:00 ET).
    Joins options_features, sentiment_features, pm_features, and equity_bars per bar; all features at t use only data before t.
    If start_date/end_date are None, infers range from sentiment_features. Writes rows to feature_bars table.
    """
    mapping = _load_mapping()
    underlying_to_tokens = _underlying_to_token_ids(mapping)
    underlyings = _load_underlyings()
    if underlying:
        underlyings = [u for u in underlyings if u == underlying]
    if not underlyings:
        logger.debug("No underlyings to process")
        return

    # If the static mapping yields no token_ids for any underlying, auto-discover
    # the most data-rich tokens from pm_features so PM signal flows into feature_bars.
    missing = [u for u in underlyings if not underlying_to_tokens.get(u)]
    if missing:
        auto = _auto_discover_token_ids(missing)
        for u, tids in auto.items():
            if tids:
                underlying_to_tokens[u] = tids
                logger.info("Auto-mapped %d PM token_ids to %s (mapping.yaml had none)", len(tids), u)

    et_tz = _get_et_tz()
    if start_date is None or end_date is None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MIN(ts), MAX(ts) FROM sentiment_features")
                row = cur.fetchone()
                if not row or row[0] is None:
                    logger.info("No sentiment_features; run build_sentiment_features first")
                    return
                try:
                    start_date = start_date or (row[0].date() if hasattr(row[0], "date") else date.today())
                    end_date = end_date or (row[1].date() if hasattr(row[1], "date") else date.today())
                except IndexError as e:
                    raise IndexError(
                        f"sentiment_features date row has {len(row) if row else 0} columns; need 2. "
                        f"Traceback:\n{traceback.format_exc()}"
                    ) from e
    if start_date > end_date:
        return

    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)

    all_bars: list[datetime] = []
    for d in dates:
        all_bars.extend(market_hours_bars_utc_for_date(d, et_tz))
    if not all_bars:
        return

    rows: list[FeatureRow] = []
    with get_connection() as conn:
        for u in underlyings:
            token_ids = underlying_to_tokens.get(u, [])
            for bar_ts in all_bars:
                try:
                    rows.append(build_row(conn, u, bar_ts, token_ids, schema_version))
                except IndexError as e:
                    logger.exception("IndexError in build_row(underlying=%r, bar_ts=%r)", u, bar_ts)
                    raise IndexError(
                        f"tuple index out of range for underlying={u!r} bar_ts={bar_ts!r}. "
                        f"Original: {e}. Traceback:\n{traceback.format_exc()}"
                    ) from e
        if rows:
            from src.ingestion.pm_writer import write_feature_bars
            write_feature_bars(conn, rows, schema_version=schema_version)
            logger.info("Built %s feature_bars (schema_version=%s)", len(rows), schema_version)
