"""Tests for sentiment_features builder (15-min bars, recency/engagement/source-weighted)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.features.sentiment_features import (
    _is_macro_doc,
    _recency_weight,
    compute_features_for_bar,
    market_hours_bars_utc_for_date,
    run_build_sentiment_features,
)
from src.utils.schemas import SentimentFeature


def test_market_hours_bars_utc_for_date():
    try:
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et_tz = pytz.timezone("America/New_York")
    bars = market_hours_bars_utc_for_date(date(2025, 2, 15), et_tz)
    # 9:30 to 16:00 ET, 15-min steps: 9:30, 9:45, ..., 15:45, 16:00 -> 27 bars
    assert len(bars) == 27
    assert bars[0].hour != bars[-1].hour or bars[0].minute != bars[-1].minute


def test_is_macro_doc():
    assert _is_macro_doc("Fed raises rates amid inflation", ["inflation", "Federal Reserve"]) is True
    assert _is_macro_doc("Apple stock rises", ["inflation"]) is False
    assert _is_macro_doc("", ["CPI"]) is False
    assert _is_macro_doc("CPI report due", []) is False


def test_recency_weight():
    bar = datetime(2025, 2, 15, 15, 0, 0, tzinfo=timezone.utc)
    doc_same = datetime(2025, 2, 15, 15, 0, 0, tzinfo=timezone.utc)
    doc_4h_ago = datetime(2025, 2, 15, 11, 0, 0, tzinfo=timezone.utc)
    assert _recency_weight(doc_same, bar, 4.0) == pytest.approx(1.0)
    assert _recency_weight(doc_4h_ago, bar, 4.0) == pytest.approx(0.5, abs=0.01)  # one half-life
    assert _recency_weight(doc_same, bar, 0) == 1.0


def test_compute_features_for_bar_no_docs():
    bar = datetime(2025, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    feat = compute_features_for_bar(
        "SPY",
        bar,
        [],
        ["inflation", "CPI"],
        4.0,
        4,
        None,
        1.0,
        0.5,
    )
    assert feat.no_news_flag is True
    assert feat.sent_news_asset == 0.0
    assert feat.sent_social_asset == 0.0
    assert feat.sent_volume == 0
    assert feat.underlying == "SPY"
    assert feat.ts == bar


def test_compute_features_for_bar_with_docs():
    bar = datetime(2025, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    doc_ts = datetime(2025, 2, 15, 9, 50, 0, tzinfo=timezone.utc)
    # (id, source, ts, text, tickers, engagement, compound, model)
    docs = [
        ("1", "newsapi", doc_ts, "SPY gains on inflation data.", ["SPY"], None, 0.5, "finbert"),
        ("2", "newsapi", doc_ts, "CPI report.", ["SPY"], None, -0.2, "finbert"),
        ("3", "reddit", doc_ts, "Markets up.", ["SPY"], 10.0, 0.3, "vader"),
    ]
    feat = compute_features_for_bar(
        "SPY",
        bar,
        docs,
        ["inflation", "CPI"],
        4.0,
        4,
        None,
        1.0,
        0.5,
    )
    assert feat.no_news_flag is False
    assert feat.sent_volume == 3
    assert feat.sent_news_asset != 0.0  # recency-weighted mean of 0.5 and -0.2
    assert feat.sent_social_asset == pytest.approx(0.3)  # single social, engagement-weighted
    assert feat.sent_macro_topic != 0.0  # first two docs have macro keywords
    assert feat.sent_dispersion >= 0
    assert feat.sent_momentum == 0.0  # prev_mean was None


def test_compute_features_for_bar_momentum():
    bar = datetime(2025, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    doc_ts = datetime(2025, 2, 15, 9, 50, 0, tzinfo=timezone.utc)
    docs = [
        ("1", "newsapi", doc_ts, "SPY up.", ["SPY"], None, 0.6, "finbert"),
    ]
    feat = compute_features_for_bar(
        "SPY",
        bar,
        docs,
        [],
        4.0,
        4,
        prev_bars_mean_compound=0.2,
        macro_news_weight=1.0,
        macro_social_weight=0.5,
    )
    assert feat.sent_momentum == pytest.approx(0.6 - 0.2)


def test_run_build_sentiment_features_no_data():
    with patch("src.features.sentiment_features.get_connection") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (None, None)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__.return_value = None
        run_build_sentiment_features(start_date=date(2025, 2, 15), end_date=date(2025, 2, 15))
    # Should return without writing (no sentiment_scored data)
    mock_cur.execute.assert_called()


def test_run_build_sentiment_features_with_mocked_docs():
    with patch("src.features.sentiment_features.get_connection") as mock_conn:
        with patch("src.features.sentiment_features._load_scored_docs") as mock_load:
            mock_load.return_value = []
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = (
                datetime(2025, 2, 15, 14, 30, tzinfo=timezone.utc),
                datetime(2025, 2, 15, 20, 0, tzinfo=timezone.utc),
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__.return_value = None
            run_build_sentiment_features(
                underlying="SPY",
                start_date=date(2025, 2, 15),
                end_date=date(2025, 2, 15),
            )
    # Should have called write_sentiment_features with a list of features (possibly empty or with no_news rows)
    mock_conn.return_value.__enter__.assert_called()


def test_write_sentiment_features_calls_cursor():
    from src.ingestion.sentiment_writer import write_sentiment_features
    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    row = SentimentFeature(
        underlying="SPY",
        ts=datetime(2025, 2, 15, 14, 30, tzinfo=timezone.utc),
        sent_news_asset=0.1,
        sent_social_asset=0.0,
        sent_macro_topic=-0.05,
        sent_dispersion=0.01,
        sent_momentum=0.02,
        sent_volume=5,
        no_news_flag=False,
    )
    write_sentiment_features(mock_conn, [row], schema_version=1)
    assert mock_cur.execute.called
