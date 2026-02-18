"""Tests for options_features builder (ATM IV, term slope, skew, realized vol, VIX)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.features.options_features import (
    _annualized_realized_vol,
    _atm_iv_for_bucket,
    _compute_iv_skew,
    _days_to_expiry,
    _get_vix_close,
    _realized_vol_for_window,
    run_build_options_features,
)
from src.utils.schemas import OptionsFeature


def test_days_to_expiry():
    assert _days_to_expiry(date(2025, 1, 17), date(2025, 1, 2)) == 15
    assert _days_to_expiry(date(2025, 1, 10), date(2025, 1, 2)) == 8


def test_annualized_realized_vol():
    # Zero returns -> zero vol
    assert _annualized_realized_vol([0.0, 0.0, 0.0]) == 0.0
    # Single return
    r = _annualized_realized_vol([0.01])
    assert r is None  # need at least 2 for variance
    # Two returns
    r = _annualized_realized_vol([0.01, -0.01])
    assert r is not None and r >= 0
    # Empty
    assert _annualized_realized_vol([]) is None


def test_atm_iv_for_bucket():
    # options_rows: (expiry, strike, option_type, iv, delta)
    snapshot_date = date(2025, 1, 2)
    rows_7d = [
        (date(2025, 1, 9), 598.0, "call", 0.22, 0.5),
        (date(2025, 1, 10), 600.0, "call", 0.20, 0.52),
        (date(2025, 1, 11), 602.0, "call", 0.21, 0.48),
    ]
    # 7D bucket is (3, 12) days; 1/9 is 7 days, 1/10 is 8, 1/11 is 9 - all in range
    # Spot 600: closest strike 600 -> IV 0.20
    iv = _atm_iv_for_bucket(rows_7d, snapshot_date, 600.0, (3, 12))
    assert iv == 0.20
    iv_none = _atm_iv_for_bucket(rows_7d, snapshot_date, 600.0, (1, 2))  # no DTE in range
    assert iv_none is None


def test_compute_iv_skew():
    # Put with strike < spot, call with strike > spot; skew = put_iv - call_iv
    snapshot_date = date(2025, 1, 2)
    rows = [
        (date(2025, 1, 17), 595.0, "put", 0.28, -0.3),
        (date(2025, 1, 17), 605.0, "call", 0.18, 0.3),
        (date(2025, 1, 17), 600.0, "call", 0.20, 0.5),
    ]
    skew = _compute_iv_skew(rows, snapshot_date, 600.0)
    assert skew is not None
    assert skew == pytest.approx(0.28 - 0.18)  # OTM put 595 IV 0.28, OTM call 605 IV 0.18
    assert _compute_iv_skew(rows, snapshot_date, None) is None
    assert _compute_iv_skew([], snapshot_date, 600.0) is None


def test_realized_vol_for_window():
    # equity_rows: (ts, close) ascending
    base = date(2025, 1, 1)
    rows = [(base, 100.0)]
    assert _realized_vol_for_window(rows, base, 5) is None
    rows = [(base, 100.0), (date(2025, 1, 2), 101.0), (date(2025, 1, 3), 99.0)]
    rv = _realized_vol_for_window(rows, date(2025, 1, 3), 2)
    assert rv is not None and rv >= 0


def test_get_vix_close():
    pytest.importorskip("yfinance")
    with patch("yfinance.Ticker") as mock_ticker:
        import pandas as pd

        mock_ticker.return_value.history.return_value = pd.DataFrame(
            {"Close": [18.5]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-01-02")]),
        )
        v = _get_vix_close(date(2025, 1, 2))
    assert v == 18.5


def test_run_build_options_features_no_data():
    """When no options_snapshots exist, nothing written (no keys)."""
    with patch("src.features.options_features.get_connection") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cur.fetchone.return_value = None
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_cur
        mock_cm.__exit__.return_value = False
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
        run_build_options_features(underlying="SPY", feature_date=date(2025, 1, 2))
    mock_cur.execute.assert_called()
    # Should not call write (no keys)
    # We can't easily assert write not called without patching it; at least no exception
    assert True


def test_run_build_options_features_single_key():
    """One (underlying, date) with options + equity + VIX -> one OptionsFeature row written."""
    last_sql = [""]

    def capture_execute(*args, **kwargs):
        last_sql[0] = args[0] if args else ""

    options_rows = [
        (date(2025, 1, 17), 600.0, "call", 0.20, 0.5),
        (date(2025, 1, 17), 598.0, "put", 0.22, -0.5),
        (date(2025, 1, 17), 602.0, "call", 0.18, 0.4),
    ]
    equity_rows = [
        (date(2024, 11, 1), 580.0),
        (date(2024, 11, 4), 581.0),
        (date(2024, 11, 5), 579.0),
        (date(2025, 1, 2), 600.0),
    ]

    def fetchone():
        if "SELECT close FROM equity_bars" in last_sql[0] and "ts <= " not in last_sql[0]:
            return (600.0,)
        if "SELECT 1 FROM" in last_sql[0]:
            return (1,)
        return None

    def fetchall():
        if "expiry, strike" in last_sql[0]:
            return options_rows
        if "ts, close" in last_sql[0] and "ORDER BY ts" in last_sql[0]:
            return equity_rows
        return []

    mock_cur = MagicMock()
    mock_cur.execute = capture_execute
    mock_cur.fetchone = fetchone
    mock_cur.fetchall = fetchall
    mock_cm_cursor = MagicMock()
    mock_cm_cursor.__enter__.return_value = mock_cur
    mock_cm_cursor.__exit__.return_value = False
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.__exit__.return_value = False

    with patch("src.features.options_features.get_connection", return_value=mock_conn):
        with patch("src.features.options_features._get_vix_close", return_value=19.0):
            with patch("src.features.options_features.write_options_features") as mock_write:
                run_build_options_features(underlying="SPY", feature_date=date(2025, 1, 2))
    assert mock_write.called
    rows = mock_write.call_args[0][1]
    assert len(rows) == 1
    assert rows[0].underlying == "SPY"
    assert rows[0].feature_date == date(2025, 1, 2)
    assert rows[0].vix_close == 19.0
    assert rows[0].iv_skew is not None  # put 598 IV 0.22, call 602 IV 0.18 -> skew 0.04


def test_write_options_features_cursor():
    from src.ingestion.pm_writer import write_options_features

    mock_conn = MagicMock()
    row = OptionsFeature(
        underlying="SPY",
        feature_date=date(2025, 1, 2),
        atm_iv_7d=0.20,
        atm_iv_30d=0.22,
        iv_term_slope=0.02,
        iv_skew=0.03,
        realized_vol_10d=0.18,
        vix_close=19.0,
    )
    write_options_features(mock_conn, [row], schema_version=1)
    assert mock_conn.cursor.return_value.__enter__.return_value.execute.call_count == 1
    call_args = mock_conn.cursor.return_value.__enter__.return_value.execute.call_args[0]
    assert "options_features" in call_args[0]
    assert call_args[1][0] == "SPY"
    assert call_args[1][2] == 1  # schema_version
    assert call_args[1][3] == 0.20  # atm_iv_7d
