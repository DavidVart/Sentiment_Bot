"""Build pm_features from pm_prices: p, logit_p, delta_p_1h, delta_p_1d, vol_of_p, time_to_event, surprise_z."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from src.db import get_connection
from src.ingestion.pm_writer import write_pm_features
from src.utils.logging_utils import get_logger
from src.utils.schemas import PMFeature

logger = get_logger(__name__)


def _logit(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return math.log(p / (1 - p))


def run_build_pm_features(token_id: str | None = None, schema_version: int = 1) -> None:
    """
    Read pm_prices for each token_id, compute derived features, upsert into pm_features.
    Features: p, logit_p, delta_p_1h, delta_p_1d, vol_of_p (rolling std), time_to_event, surprise_z.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if token_id:
                cur.execute("SELECT DISTINCT token_id FROM pm_prices WHERE token_id = %s", (token_id,))
            else:
                cur.execute("SELECT DISTINCT token_id FROM pm_prices ORDER BY token_id")
            token_ids = [row[0] for row in cur.fetchall()]
    for tid in token_ids:
        _build_for_token(tid, schema_version)


def _build_for_token(token_id: str, schema_version: int) -> None:
    """Load prices for token_id, compute features, write pm_features."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts, price FROM pm_prices WHERE token_id = %s ORDER BY ts",
                (token_id,),
            )
            rows = cur.fetchall()
    if not rows:
        logger.debug("No prices for token_id=%s", token_id[:20])
        return
    # Build (ts, price) sorted by ts
    ts_list = [r[0] for r in rows]
    p_list = [float(r[1]) for r in rows]
    one_hour = timedelta(hours=1)
    one_day = timedelta(days=1)
    window_vol = 24  # 24 points for rolling vol (e.g. 24h if hourly)
    features: list[PMFeature] = []
    for i, (ts, p) in enumerate(zip(ts_list, p_list)):
        logit_p = _logit(p)
        # delta_p_1h: price now - price 1h ago (approx: previous point if 15m bars = 4 points)
        delta_p_1h = None
        for j in range(i - 1, -1, -1):
            if ts_list[j] <= ts - one_hour:
                delta_p_1h = p - p_list[j]
                break
        delta_p_1d = None
        for j in range(i - 1, -1, -1):
            if ts_list[j] <= ts - one_day:
                delta_p_1d = p - p_list[j]
                break
        vol_of_p = None
        if i >= window_vol:
            window_p = p_list[i - window_vol : i]
            vol_of_p = float(math.sqrt(sum((x - sum(window_p) / len(window_p)) ** 2 for x in window_p) / len(window_p)))
        time_to_event = None  # would need event end_ts from pm_events; skip for MVP
        surprise_z = None
        if vol_of_p and vol_of_p > 0 and delta_p_1h is not None:
            surprise_z = delta_p_1h / vol_of_p
        features.append(
            PMFeature(
                token_id=token_id,
                ts=ts,
                p=p,
                logit_p=logit_p,
                delta_p_1h=delta_p_1h,
                delta_p_1d=delta_p_1d,
                momentum=delta_p_1h,
                vol_of_p=vol_of_p,
                time_to_event=time_to_event,
                surprise_z=surprise_z,
            )
        )
    if features:
        with get_connection() as conn:
            write_pm_features(conn, features, schema_version=schema_version)
        logger.info("Built %s pm_features for token_id=%s", len(features), token_id[:20])
