"""Write normalized prediction-market entities to PostgreSQL."""

from __future__ import annotations

from datetime import datetime

from psycopg2.extensions import connection as PgConnection

from src.db import get_connection
from src.utils.logging_utils import get_logger
from src.utils.schemas import EquityBar, FeatureRow, OptionsFeature, OptionsSnapshot, PMEvent, PMFeature, PMMarket, PMPrice

logger = get_logger(__name__)


def _ts(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def write_pm_events(conn: PgConnection, events: list[PMEvent]) -> None:
    """Upsert pm_events (on conflict update)."""
    with conn.cursor() as cur:
        for e in events:
            cur.execute(
                """
                INSERT INTO pm_events (event_id, platform, title, category, start_ts, end_ts, status, resolution_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    category = EXCLUDED.category,
                    start_ts = EXCLUDED.start_ts,
                    end_ts = EXCLUDED.end_ts,
                    status = EXCLUDED.status,
                    resolution_ts = EXCLUDED.resolution_ts
                """,
                (
                    e.event_id,
                    e.platform,
                    e.title,
                    e.category,
                    _ts(e.start_ts),
                    _ts(e.end_ts),
                    e.status,
                    _ts(e.resolution_ts),
                ),
            )
    logger.info("Wrote %s pm_events", len(events))


def write_pm_markets(conn: PgConnection, markets: list[PMMarket]) -> None:
    """Upsert pm_markets."""
    with conn.cursor() as cur:
        for m in markets:
            cur.execute(
                """
                INSERT INTO pm_markets (market_id, event_id, platform, slug, outcome_names, token_ids, active, volume, liquidity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (market_id) DO UPDATE SET
                    event_id = EXCLUDED.event_id,
                    slug = EXCLUDED.slug,
                    outcome_names = EXCLUDED.outcome_names,
                    token_ids = EXCLUDED.token_ids,
                    active = EXCLUDED.active,
                    volume = EXCLUDED.volume,
                    liquidity = EXCLUDED.liquidity
                """,
                (
                    m.market_id,
                    m.event_id,
                    m.platform,
                    m.slug,
                    m.outcome_names,
                    m.token_ids,
                    m.active,
                    m.volume,
                    m.liquidity,
                ),
            )
    logger.info("Wrote %s pm_markets", len(markets))


def write_pm_prices(conn: PgConnection, prices: list[PMPrice], batch_size: int = 500) -> None:
    """Insert pm_prices (no conflict; append-only)."""
    with conn.cursor() as cur:
        for i in range(0, len(prices), batch_size):
            batch = prices[i : i + batch_size]
            for p in batch:
                cur.execute(
                    """
                    INSERT INTO pm_prices (token_id, platform, ts, price, mid, best_bid, best_ask, spread, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        p.token_id,
                        p.platform,
                        p.ts,
                        p.price,
                        p.mid,
                        p.best_bid,
                        p.best_ask,
                        p.spread,
                        p.source,
                    ),
                )
    logger.info("Wrote %s pm_prices", len(prices))


def write_pm_features(conn: PgConnection, features: list[PMFeature], schema_version: int = 1) -> None:
    """Upsert pm_features (unique on token_id, ts)."""
    with conn.cursor() as cur:
        for f in features:
            cur.execute(
                """
                INSERT INTO pm_features (token_id, ts, schema_version, p, logit_p, delta_p_1h, delta_p_1d, momentum, vol_of_p, time_to_event, surprise_z)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (token_id, ts) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    p = EXCLUDED.p,
                    logit_p = EXCLUDED.logit_p,
                    delta_p_1h = EXCLUDED.delta_p_1h,
                    delta_p_1d = EXCLUDED.delta_p_1d,
                    momentum = EXCLUDED.momentum,
                    vol_of_p = EXCLUDED.vol_of_p,
                    time_to_event = EXCLUDED.time_to_event,
                    surprise_z = EXCLUDED.surprise_z
                """,
                (
                    f.token_id,
                    f.ts,
                    schema_version,
                    f.p,
                    f.logit_p,
                    f.delta_p_1h,
                    f.delta_p_1d,
                    f.momentum,
                    f.vol_of_p,
                    f.time_to_event,
                    f.surprise_z,
                ),
            )
    logger.info("Wrote %s pm_features", len(features))


def write_equity_bars(conn: PgConnection, bars: list[EquityBar]) -> None:
    """Upsert equity_bars (symbol, ts)."""
    with conn.cursor() as cur:
        for b in bars:
            cur.execute(
                """
                INSERT INTO equity_bars (symbol, ts, open, high, low, close, volume, vwap, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap,
                    source = EXCLUDED.source
                """,
                (
                    b.symbol,
                    b.ts,
                    b.open,
                    b.high,
                    b.low,
                    b.close,
                    b.volume,
                    b.vwap,
                    b.source,
                ),
            )
    logger.info("Wrote %s equity_bars", len(bars))


def write_options_snapshots(conn: PgConnection, rows: list[OptionsSnapshot]) -> None:
    """Upsert options_snapshots (underlying, snapshot_date, contract_id)."""
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO options_snapshots (
                    underlying, snapshot_date, contract_id, expiry, strike, option_type,
                    bid, ask, mid, close, iv, delta, gamma, theta, vega,
                    volume, open_interest, source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (underlying, snapshot_date, contract_id) DO UPDATE SET
                    expiry = EXCLUDED.expiry,
                    strike = EXCLUDED.strike,
                    option_type = EXCLUDED.option_type,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    mid = EXCLUDED.mid,
                    close = EXCLUDED.close,
                    iv = EXCLUDED.iv,
                    delta = EXCLUDED.delta,
                    gamma = EXCLUDED.gamma,
                    theta = EXCLUDED.theta,
                    vega = EXCLUDED.vega,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    source = EXCLUDED.source
                """,
                (
                    row.underlying,
                    row.snapshot_date,
                    row.contract_id,
                    row.expiry,
                    row.strike,
                    row.option_type,
                    row.bid,
                    row.ask,
                    row.mid,
                    row.close,
                    row.iv,
                    row.delta,
                    row.gamma,
                    row.theta,
                    row.vega,
                    row.volume,
                    row.open_interest,
                    row.source,
                ),
            )
    logger.info("Wrote %s options_snapshots", len(rows))


def write_options_features(
    conn: PgConnection,
    rows: list[OptionsFeature],
    schema_version: int = 1,
) -> None:
    """Upsert options_features (underlying, feature_date)."""
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO options_features (
                    underlying, feature_date, schema_version,
                    atm_iv_7d, atm_iv_14d, atm_iv_30d, iv_term_slope, iv_skew,
                    realized_vol_5d, realized_vol_10d, realized_vol_20d, realized_vol_60d,
                    vix_close
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (underlying, feature_date) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    atm_iv_7d = EXCLUDED.atm_iv_7d,
                    atm_iv_14d = EXCLUDED.atm_iv_14d,
                    atm_iv_30d = EXCLUDED.atm_iv_30d,
                    iv_term_slope = EXCLUDED.iv_term_slope,
                    iv_skew = EXCLUDED.iv_skew,
                    realized_vol_5d = EXCLUDED.realized_vol_5d,
                    realized_vol_10d = EXCLUDED.realized_vol_10d,
                    realized_vol_20d = EXCLUDED.realized_vol_20d,
                    realized_vol_60d = EXCLUDED.realized_vol_60d,
                    vix_close = EXCLUDED.vix_close
                """,
                (
                    row.underlying,
                    row.feature_date,
                    schema_version,
                    row.atm_iv_7d,
                    row.atm_iv_14d,
                    row.atm_iv_30d,
                    row.iv_term_slope,
                    row.iv_skew,
                    row.realized_vol_5d,
                    row.realized_vol_10d,
                    row.realized_vol_20d,
                    row.realized_vol_60d,
                    row.vix_close,
                ),
            )
    logger.info("Wrote %s options_features", len(rows))


def write_feature_bars(
    conn: PgConnection,
    rows: list[FeatureRow],
    schema_version: int = 1,
) -> None:
    """Upsert feature_bars (underlying, ts). Schema version must match; bump for full recompute."""
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO feature_bars (
                    underlying, ts, schema_version,
                    atm_iv_7d, atm_iv_14d, atm_iv_30d, iv_term_slope, iv_skew,
                    realized_vol_5d, realized_vol_10d, realized_vol_20d, realized_vol_60d, vix_close,
                    options_gap_flag,
                    sent_news_asset, sent_social_asset, sent_macro_topic, sent_dispersion, sent_momentum, sent_volume, no_news_flag,
                    pm_p, pm_logit_p, pm_delta_p_1h, pm_delta_p_1d, pm_momentum, pm_vol_of_p, pm_time_to_event, pm_surprise_z, pm_gap_flag,
                    equity_return_1d, equity_realized_vol_20d
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (underlying, ts) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    atm_iv_7d = EXCLUDED.atm_iv_7d,
                    atm_iv_14d = EXCLUDED.atm_iv_14d,
                    atm_iv_30d = EXCLUDED.atm_iv_30d,
                    iv_term_slope = EXCLUDED.iv_term_slope,
                    iv_skew = EXCLUDED.iv_skew,
                    realized_vol_5d = EXCLUDED.realized_vol_5d,
                    realized_vol_10d = EXCLUDED.realized_vol_10d,
                    realized_vol_20d = EXCLUDED.realized_vol_20d,
                    realized_vol_60d = EXCLUDED.realized_vol_60d,
                    vix_close = EXCLUDED.vix_close,
                    options_gap_flag = EXCLUDED.options_gap_flag,
                    sent_news_asset = EXCLUDED.sent_news_asset,
                    sent_social_asset = EXCLUDED.sent_social_asset,
                    sent_macro_topic = EXCLUDED.sent_macro_topic,
                    sent_dispersion = EXCLUDED.sent_dispersion,
                    sent_momentum = EXCLUDED.sent_momentum,
                    sent_volume = EXCLUDED.sent_volume,
                    no_news_flag = EXCLUDED.no_news_flag,
                    pm_p = EXCLUDED.pm_p,
                    pm_logit_p = EXCLUDED.pm_logit_p,
                    pm_delta_p_1h = EXCLUDED.pm_delta_p_1h,
                    pm_delta_p_1d = EXCLUDED.pm_delta_p_1d,
                    pm_momentum = EXCLUDED.pm_momentum,
                    pm_vol_of_p = EXCLUDED.pm_vol_of_p,
                    pm_time_to_event = EXCLUDED.pm_time_to_event,
                    pm_surprise_z = EXCLUDED.pm_surprise_z,
                    pm_gap_flag = EXCLUDED.pm_gap_flag,
                    equity_return_1d = EXCLUDED.equity_return_1d,
                    equity_realized_vol_20d = EXCLUDED.equity_realized_vol_20d
                """,
                (
                    row.underlying,
                    _ts(row.ts),
                    schema_version,
                    row.atm_iv_7d,
                    row.atm_iv_14d,
                    row.atm_iv_30d,
                    row.iv_term_slope,
                    row.iv_skew,
                    row.realized_vol_5d,
                    row.realized_vol_10d,
                    row.realized_vol_20d,
                    row.realized_vol_60d,
                    row.vix_close,
                    row.options_gap_flag,
                    row.sent_news_asset,
                    row.sent_social_asset,
                    row.sent_macro_topic,
                    row.sent_dispersion,
                    row.sent_momentum,
                    row.sent_volume,
                    row.no_news_flag,
                    row.pm_p,
                    row.pm_logit_p,
                    row.pm_delta_p_1h,
                    row.pm_delta_p_1d,
                    row.pm_momentum,
                    row.pm_vol_of_p,
                    row.pm_time_to_event,
                    row.pm_surprise_z,
                    row.pm_gap_flag,
                    row.equity_return_1d,
                    row.equity_realized_vol_20d,
                ),
            )
    logger.info("Wrote %s feature_bars", len(rows))
