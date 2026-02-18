"""Tests for equity provider (Polygon + yfinance) and equity backfill."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.connectors.marketdata import (
    PolygonEquityProvider,
    YFinanceEquityProvider,
    get_equity_provider,
)
from src.utils.rate_limit import RateLimiter
from src.utils.schemas import EquityBar


def test_rate_limiter_interval():
    limiter = RateLimiter(calls_per_minute=5)
    assert limiter._interval == 12.0
    limiter.wait_if_needed()
    limiter.wait_if_needed()  # second call should block ~12s or not if tested quickly


def test_polygon_parse_aggregates():
    """Polygon provider normalizes /v2/aggs response to EquityBar list."""
    provider = PolygonEquityProvider(api_key="test-key")
    raw = {
        "results": [
            {
                "t": 1704067200000,  # 2024-01-01 UTC
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1_000_000,
                "vw": 100.5,
            },
            {
                "t": 1704153600000,
                "o": 101.0,
                "h": 103.0,
                "l": 100.0,
                "c": 102.0,
                "v": 1_100_000,
                "vw": 101.2,
            },
        ]
    }

    with patch.object(provider, "_get", return_value=raw):
        bars = provider.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(bars) == 2
    for b in bars:
        assert isinstance(b, EquityBar)
        assert b.symbol == "SPY"
        assert b.source == "polygon"
    assert bars[0].open == 100.0 and bars[0].close == 101.0 and bars[0].vwap == 100.5
    assert bars[1].volume == 1_100_000


def test_polygon_empty_results():
    provider = PolygonEquityProvider(api_key="test-key")
    with patch.object(provider, "_get", return_value={"results": []}):
        bars = provider.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert bars == []


def test_polygon_no_key_warns():
    provider = PolygonEquityProvider(api_key="")
    with patch.object(provider, "_get", side_effect=Exception("unauthorized")):
        bars = provider.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert bars == []


def test_yfinance_fetch_daily_bars():
    """yfinance provider returns EquityBar list from history."""
    pytest.importorskip("yfinance")
    import pandas as pd

    provider = YFinanceEquityProvider()
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1e6, 1.1e6],
        },
        index=idx,
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = df
        bars = provider.fetch_daily_bars("QQQ", date(2024, 1, 1), date(2024, 1, 10))
    assert len(bars) == 2
    for b in bars:
        assert isinstance(b, EquityBar)
        assert b.symbol == "QQQ"
        assert b.source == "yfinance"
    assert bars[0].open == 100.0 and bars[0].close == 101.0
    assert bars[1].volume == 1.1e6


def test_yfinance_empty_history():
    pytest.importorskip("yfinance")
    provider = YFinanceEquityProvider()
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = None
        bars = provider.fetch_daily_bars("INVALID", date(2024, 1, 1), date(2024, 1, 5))
    assert bars == []


def test_get_equity_provider_polygon_when_key_set():
    with patch.dict("os.environ", {"MASSIVE_API": "key123"}, clear=False):
        p = get_equity_provider(prefer="polygon")
    assert isinstance(p, PolygonEquityProvider)
    assert p.api_key == "key123"


def test_get_equity_provider_yfinance_when_no_key():
    pytest.importorskip("yfinance")
    with patch.dict("os.environ", {"MASSIVE_API": ""}, clear=False):
        p = get_equity_provider(prefer="polygon")
    # When key is empty, falls back to yfinance
    assert isinstance(p, YFinanceEquityProvider)


def test_write_equity_bars_calls_cursor():
    from src.ingestion.pm_writer import write_equity_bars

    mock_conn = MagicMock()
    bars = [
        EquityBar(
            symbol="SPY",
            ts=date(2024, 1, 2),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1e6,
            vwap=100.5,
            source="polygon",
        ),
    ]
    write_equity_bars(mock_conn, bars)
    assert mock_conn.cursor.return_value.__enter__.return_value.execute.call_count == 1
    call_args = mock_conn.cursor.return_value.__enter__.return_value.execute.call_args[0]
    assert "equity_bars" in call_args[0]
    assert call_args[1][0] == "SPY"
    assert call_args[1][2] == 100.0
