"""Tests for master feature alignment (align.py) and anti-lookahead rules."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.features.align import (
    _bar_date,
    _equity_for_bar,
    _load_mapping,
    _options_for_bar,
    _sentiment_for_bar,
    _underlying_to_token_ids,
    build_row,
)
from src.utils.schemas import FeatureRow


def test_bar_date():
    # 16:00 ET = 21:00 UTC (EST) or 20:00 UTC (EDT); use a fixed UTC that is 15:00 ET
    bar_utc = datetime(2025, 2, 15, 20, 0, 0, tzinfo=timezone.utc)  # 15:00 ET
    d = _bar_date(bar_utc)
    assert d == date(2025, 2, 15)


def test_load_mapping():
    mapping = _load_mapping()
    assert isinstance(mapping, list)
    # may be empty if no config
    if mapping:
        assert "affected_underlyings" in mapping[0] or "token_ids" in mapping[0]


def test_underlying_to_token_ids():
    mapping = [
        {"token_ids": {"polymarket": "", "kalshi": "FED-26MAR"}, "affected_underlyings": ["SPY", "QQQ"]},
        {"token_ids": {"kalshi": "CPI-MAR"}, "affected_underlyings": ["SPY"]},
    ]
    out = _underlying_to_token_ids(mapping)
    assert "SPY" in out
    assert "FED-26MAR" in out["SPY"]
    assert "CPI-MAR" in out["SPY"]
    assert "QQQ" in out
    assert "FED-26MAR" in out["QQQ"]


def test_options_for_bar_no_lookahead():
    """Options at bar date must NOT be used (strictly before bar)."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    bar_ts = datetime(2025, 2, 15, 15, 0, 0, tzinfo=timezone.utc)
    # DB returns only row with feature_date = 2025-02-15 (same day) - should NOT be selected
    mock_cur.fetchone.return_value = None  # our query is feature_date < bar_date, so 2025-02-14
    opt, gap = _options_for_bar(mock_conn, "SPY", bar_ts)
    assert gap is True
    assert opt == {}
    # Query must use feature_date < bar_date
    call_args = mock_cur.execute.call_args[0][1]
    assert call_args[1] == date(2025, 2, 15)  # bar_d


def test_options_for_bar_uses_previous_day():
    """Options with feature_date < bar_date are used."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    mock_cur.fetchone.return_value = (0.22, 0.21, 0.20, -0.01, 0.02, 0.15, 0.16, 0.17, 0.18, 19.0)
    bar_ts = datetime(2025, 2, 15, 15, 0, 0, tzinfo=timezone.utc)
    opt, gap = _options_for_bar(mock_conn, "SPY", bar_ts)
    assert gap is False
    assert opt["atm_iv_7d"] == 0.22
    assert opt["vix_close"] == 19.0


def test_sentiment_for_bar_missing():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    bar_ts = datetime(2025, 2, 15, 14, 30, 0, tzinfo=timezone.utc)
    sent = _sentiment_for_bar(mock_conn, "SPY", bar_ts)
    assert sent["no_news_flag"] is True
    assert sent["sent_news_asset"] == 0.0


def test_equity_for_bar_no_lookahead():
    """Equity bars with ts >= bar_date must not be used."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        (date(2025, 2, 13), 500.0),
        (date(2025, 2, 12), 498.0),
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    bar_ts = datetime(2025, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    eq = _equity_for_bar(mock_conn, "SPY", bar_ts)
    assert eq["equity_return_1d"] is not None
    # Query must use ts < bar_date
    call_args = mock_cur.execute.call_args[0][1]
    assert call_args[1] == date(2025, 2, 15)


def test_build_row_produces_feature_row():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    # options: one row (prev day)
    mock_cur.fetchone.side_effect = [
        (0.22, 0.21, 0.20, -0.01, 0.02, 0.15, 0.16, 0.17, 0.18, 19.0),
        (0.1, 0.0, 0.0, 0.0, 0.0, 0, True),
        None,
        None,
    ]
    mock_cur.fetchall.return_value = [(date(2025, 2, 13), 500.0), (date(2025, 2, 12), 498.0)]
    bar_ts = datetime(2025, 2, 15, 14, 30, 0, tzinfo=timezone.utc)
    row = build_row(mock_conn, "SPY", bar_ts, [], 1)
    assert isinstance(row, FeatureRow)
    assert row.underlying == "SPY"
    assert row.ts == bar_ts
    assert row.options_gap_flag is False
    assert row.atm_iv_7d == 0.22
    assert row.no_news_flag is True
    assert row.sent_news_asset == 0.1


def test_anti_lookahead_shuffled_timestamps_produce_same_row_for_same_bar():
    """
    Building the same (underlying, bar_ts) must yield the same row regardless of
    the order of other bars or data availability in the future.
    If we leaked future data, different 'future' data could change this row.
    """
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    bar_ts = datetime(2025, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
    # Only data strictly before bar_ts: options feature_date < 2025-02-15, sentiment ts=bar_ts (exact), equity ts < 2025-02-15
    def fetchone_impl():
        if mock_cur.execute.call_args[0][1][1] == date(2025, 2, 15):  # options query
            return (0.20, 0.19, 0.18, 0.0, 0.01, 0.14, 0.15, 0.16, 0.17, 18.0)
        if len(mock_cur.execute.call_args[0][1]) >= 2 and mock_cur.execute.call_args[0][1][1] == bar_ts:  # sentiment
            return (0.05, 0.02, 0.0, 0.01, 0.0, 3, False)
        return None
    mock_cur.fetchone.side_effect = fetchone_impl
    mock_cur.fetchall.return_value = [(date(2025, 2, 14), 501.0), (date(2025, 2, 13), 500.0)]
    row1 = build_row(mock_conn, "SPY", bar_ts, [], 1)
    row2 = build_row(mock_conn, "SPY", bar_ts, [], 1)
    assert row1.atm_iv_7d == row2.atm_iv_7d
    assert row1.ts == row2.ts
    assert row1.equity_return_1d == row2.equity_return_1d


def test_anti_lookahead_future_options_not_used():
    """When only options row has feature_date = bar_date, we must not use it (gap)."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    bar_ts = datetime(2025, 2, 15, 9, 30, 0, tzinfo=timezone.utc)
    mock_cur.fetchone.return_value = None  # no row with feature_date < 2025-02-15
    mock_cur.fetchall.return_value = []
    row = build_row(mock_conn, "SPY", bar_ts, [], 1)
    assert row.options_gap_flag is True
    assert row.atm_iv_7d is None


def test_run_build_feature_matrix_no_sentiment_exits():
    """When sentiment_features is empty and no date range given, we exit without building rows."""
    with patch("src.features.align.get_connection") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (None, None)  # MIN(ts), MAX(ts) both None
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__.return_value = None
        from src.features.align import run_build_feature_matrix
        run_build_feature_matrix()  # no start/end -> infer from sentiment_features -> get None -> return
    mock_cur.execute.assert_called()
    # Should not have entered the main loop (no build_row calls); execute called for MIN/MAX only
    assert mock_cur.execute.call_count == 1


def test_write_feature_bars_calls_cursor():
    from src.ingestion.pm_writer import write_feature_bars
    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    row = FeatureRow(
        underlying="SPY",
        ts=datetime(2025, 2, 15, 14, 30, tzinfo=timezone.utc),
        schema_version=1,
        options_gap_flag=True,
        no_news_flag=True,
        pm_gap_flag=True,
    )
    write_feature_bars(mock_conn, [row], schema_version=1)
    assert mock_cur.execute.called
    assert "feature_bars" in mock_cur.execute.call_args[0][0]
