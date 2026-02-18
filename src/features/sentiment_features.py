"""Build sentiment_features from sentiment_scored: 15-min bars, recency/engagement/source-weighted aggregates."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.db import get_connection
from src.ingestion.sentiment_writer import write_sentiment_features
from src.utils.logging_utils import get_logger
from src.utils.schemas import SentimentFeature

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
BAR_MINUTES = 15
MARKET_OPEN = (9, 30)  # 9:30 ET
MARKET_CLOSE = (16, 0)  # 16:00 ET (inclusive end of last bar)


def _load_sentiment_config() -> dict[str, Any]:
    path = CONFIG_DIR / "sentiment.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_underlyings() -> list[str]:
    path = CONFIG_DIR / "universe.yaml"
    if not path.exists():
        return ["SPY", "QQQ", "AAPL"]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("underlyings", ["SPY", "QQQ", "AAPL"])


def _get_et_tz() -> Any:
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        return pytz.timezone("America/New_York")


def market_hours_bars_utc_for_date(d: date, et_tz: Any) -> list[datetime]:
    """Bar end times in UTC for 15-min bars 9:30–16:00 ET on date d."""
    bars: list[datetime] = []
    hour, minute = MARKET_OPEN
    end_hour, end_minute = MARKET_CLOSE
    while (hour, minute) <= (end_hour, end_minute):
        bar_end_et = datetime(d.year, d.month, d.day, hour, minute, 0, tzinfo=et_tz)
        try:
            bar_end_utc = bar_end_et.astimezone(datetime.timezone.utc)
        except Exception:
            import pytz
            bar_end_utc = bar_end_et.astimezone(pytz.UTC)
        bars.append(bar_end_utc)
        minute += BAR_MINUTES
        if minute >= 60:
            minute -= 60
            hour += 1
    return bars


def _load_scored_docs(
    conn: Any,
    ts_start: datetime,
    ts_end: datetime,
) -> list[tuple[str, str, datetime, str, list[str], float | None, float, str]]:
    """Returns (id, source, ts, text, tickers, engagement, sentiment_compound, sentiment_model)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, ts, text, tickers, engagement, sentiment_compound, sentiment_model
            FROM sentiment_scored
            WHERE ts >= %s AND ts < %s
            ORDER BY ts
            """,
            (ts_start, ts_end),
        )
        rows = cur.fetchall()
    out: list[tuple[str, str, datetime, str, list[str], float | None, float, str]] = []
    for r in rows:
        ts = r[2]
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        out.append((
            r[0], r[1], ts, r[3] or "", list(r[4]) if r[4] else [], r[5], float(r[6]), (r[7] or "").strip(),
        ))
    return out


def _is_macro_doc(text: str, macro_keywords: list[str]) -> bool:
    if not text or not macro_keywords:
        return False
    lower = text.lower()
    return any(kw.lower() in lower for kw in macro_keywords)


def _is_news_source(source: str, model: str) -> bool:
    src = (source or "").lower()
    return src == "newsapi" or (model or "").lower() == "finbert"


def _is_social_source(source: str, model: str) -> bool:
    src = (source or "").lower()
    return "reddit" in src or (model or "").lower() == "vader"


def _recency_weight(doc_ts: datetime, bar_end_utc: datetime, half_life_hours: float) -> float:
    delta_h = (bar_end_utc - doc_ts).total_seconds() / 3600.0
    if half_life_hours <= 0:
        return 1.0
    return math.exp(-math.log(2) * delta_h / half_life_hours)


def _source_weight_news_social(source: str, model: str, news_w: float, social_w: float) -> float:
    if _is_news_source(source, model):
        return news_w
    return social_w


def compute_features_for_bar(
    underlying: str,
    bar_end_utc: datetime,
    docs_in_window: list[tuple[str, str, datetime, str, list[str], float | None, float, str]],
    macro_keywords: list[str],
    half_life_hours: float,
    momentum_window_bars: int,
    prev_bars_mean_compound: float | None,
    macro_news_weight: float,
    macro_social_weight: float,
) -> SentimentFeature:
    """Compute one SentimentFeature for (underlying, bar_end_utc) from docs in the bar window."""
    # Filter docs that mention this underlying (tickers overlap)
    def has_underlying(tickers: list[str]) -> bool:
        if not tickers:
            return False
        u = (underlying or "").upper()
        return any((t or "").upper() == u or u in (t or "").upper() for t in tickers)
    asset_docs = [d for d in docs_in_window if has_underlying(d[4])]
    all_docs = docs_in_window

    no_news = len(all_docs) == 0
    if no_news:
        return SentimentFeature(
            underlying=underlying,
            ts=bar_end_utc,
            sent_news_asset=0.0,
            sent_social_asset=0.0,
            sent_macro_topic=0.0,
            sent_dispersion=0.0,
            sent_momentum=0.0,
            sent_volume=0,
            no_news_flag=True,
        )

    # sent_news_asset: FinBERT compound, recency-weighted mean (asset docs, news only)
    news_asset = [d for d in asset_docs if _is_news_source(d[1], d[7])]
    if news_asset:
        weights = [_recency_weight(d[2], bar_end_utc, half_life_hours) for d in news_asset]
        total_w = sum(weights)
        sent_news_asset = sum(d[6] * w for d, w in zip(news_asset, weights)) / total_w if total_w > 0 else 0.0
    else:
        sent_news_asset = 0.0

    # sent_social_asset: VADER compound, engagement-weighted (asset docs, social only)
    social_asset = [d for d in asset_docs if _is_social_source(d[1], d[7])]
    if social_asset:
        eng = [d[5] if d[5] is not None and d[5] > 0 else 1.0 for d in social_asset]
        total_eng = sum(eng)
        sent_social_asset = sum(d[6] * e for d, e in zip(social_asset, eng)) / total_eng if total_eng > 0 else 0.0
    else:
        sent_social_asset = 0.0

    # sent_macro_topic: FinBERT compound for macro-tagged docs, source-weighted (news > social)
    macro_docs = [d for d in all_docs if _is_macro_doc(d[3], macro_keywords)]
    if macro_docs:
        weights = [_source_weight_news_social(d[1], d[7], macro_news_weight, macro_social_weight) for d in macro_docs]
        total_w = sum(weights)
        sent_macro_topic = sum(d[6] * w for d, w in zip(macro_docs, weights)) / total_w if total_w > 0 else 0.0
    else:
        sent_macro_topic = 0.0

    # sent_dispersion: variance of compound across sources (per-source mean, then variance of those)
    by_source: dict[str, list[float]] = {}
    for d in all_docs:
        by_source.setdefault(d[1], []).append(d[6])
    per_source_means = [sum(v) / len(v) for v in by_source.values() if v]
    if len(per_source_means) >= 2:
        mean_m = sum(per_source_means) / len(per_source_means)
        sent_dispersion = sum((x - mean_m) ** 2 for x in per_source_means) / len(per_source_means)
    else:
        sent_dispersion = 0.0

    # sent_momentum: current mean compound - previous window mean
    current_mean = sum(d[6] for d in all_docs) / len(all_docs) if all_docs else 0.0
    if prev_bars_mean_compound is not None:
        sent_momentum = current_mean - prev_bars_mean_compound
    else:
        sent_momentum = 0.0

    sent_volume = len(asset_docs)

    return SentimentFeature(
        underlying=underlying,
        ts=bar_end_utc,
        sent_news_asset=sent_news_asset,
        sent_social_asset=sent_social_asset,
        sent_macro_topic=sent_macro_topic,
        sent_dispersion=sent_dispersion,
        sent_momentum=sent_momentum,
        sent_volume=sent_volume,
        no_news_flag=False,
    )


def run_build_sentiment_features(
    underlying: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    schema_version: int = 1,
) -> None:
    """
    Compute sentiment_features from sentiment_scored for 15-min bars (9:30–16:00 ET).
    If underlying is None, process all underlyings from universe.yaml.
    If start_date/end_date are None, infer from sentiment_scored ts range.
    """
    config = _load_sentiment_config()
    fe = config.get("features", {})
    half_life_hours = float(fe.get("recency_half_life_hours", 4.0))
    momentum_window_bars = int(fe.get("momentum_window_bars", 4))
    macro_news_weight = float(fe.get("macro_news_weight", 1.0))
    macro_social_weight = float(fe.get("macro_social_weight", 0.5))
    news_queries = config.get("news_queries", {})
    macro_keywords = list(news_queries.get("macro", []))

    underlyings = _load_underlyings()
    if underlying:
        underlyings = [u for u in underlyings if u == underlying]
    if not underlyings:
        logger.debug("No underlyings to process")
        return

    et_tz = _get_et_tz()
    try:
        utc_tz = datetime.timezone.utc
    except Exception:
        import pytz
        utc_tz = pytz.UTC

    with get_connection() as conn:
        if start_date is None or end_date is None:
            with conn.cursor() as cur:
                cur.execute("SELECT MIN(ts), MAX(ts) FROM sentiment_scored")
                row = cur.fetchone()
                if not row or row[0] is None:
                    logger.info("No sentiment_scored data; skipping")
                    return
                ts_min, ts_max = row[0], row[1]
                if hasattr(ts_min, "date"):
                    start_date = start_date or ts_min.date()
                    end_date = end_date or ts_max.date()
                else:
                    start_date = start_date or date.today()
                    end_date = end_date or date.today()

    if start_date > end_date:
        logger.debug("start_date > end_date; skipping")
        return

    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)

    all_bars_utc: list[datetime] = []
    for d in dates:
        all_bars_utc.extend(market_hours_bars_utc_for_date(d, et_tz))
    if not all_bars_utc:
        return

    ts_start = min(all_bars_utc) - timedelta(minutes=BAR_MINUTES)
    ts_end = max(all_bars_utc) + timedelta(minutes=1)
    with get_connection() as conn:
        all_docs = _load_scored_docs(conn, ts_start, ts_end)

    # For each bar, get docs in [bar_end - 15m, bar_end)
    bar_duration = timedelta(minutes=BAR_MINUTES)
    features_out: list[SentimentFeature] = []
    for u in underlyings:
        prev_means: list[float] = []  # mean compound for last momentum_window_bars bars
        for i, bar_end_utc in enumerate(all_bars_utc):
            bar_start = bar_end_utc - bar_duration
            docs_in_bar = [d for d in all_docs if bar_start <= d[2] < bar_end_utc]
            prev_mean = (sum(prev_means) / len(prev_means)) if prev_means else None
            feat = compute_features_for_bar(
                u,
                bar_end_utc,
                docs_in_bar,
                macro_keywords,
                half_life_hours,
                momentum_window_bars,
                prev_mean,
                macro_news_weight,
                macro_social_weight,
            )
            features_out.append(feat)
            current_mean = sum(d[6] for d in docs_in_bar) / len(docs_in_bar) if docs_in_bar else 0.0
            prev_means.append(current_mean)
            if len(prev_means) > momentum_window_bars:
                prev_means.pop(0)

    if features_out:
        with get_connection() as conn:
            write_sentiment_features(conn, features_out, schema_version=schema_version)
        logger.info("Built %s sentiment_features", len(features_out))
