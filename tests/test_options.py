"""Tests for options provider (Polygon snapshot, Tradier chains, yfinance fallback, Greeks)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.connectors.marketdata import (
    PolygonOptionsProvider,
    TradierOptionsProvider,
    YFinanceOptionsProvider,
    get_polygon_options_provider,
    get_tradier_options_provider,
    get_yfinance_options_provider,
)
from src.connectors.marketdata.greeks import _years_to_expiry, compute_greeks
from src.connectors.marketdata.options_provider import _safe_float
from src.ingestion.pm_writer import write_options_snapshots
from src.utils.schemas import OptionsSnapshot


def test_years_to_expiry():
    assert _years_to_expiry(date(2024, 1, 1), date(2024, 1, 31)) > 0
    assert _years_to_expiry(date(2024, 1, 1), date(2025, 1, 1)) == pytest.approx(1.0, rel=0.01)
    assert _years_to_expiry(date(2024, 1, 1), date(2024, 1, 1)) > 0  # min 1e-4


def test_compute_greeks():
    pytest.importorskip("py_vollib")
    g = compute_greeks(flag="c", S=100.0, K=100.0, t=0.25, sigma=0.25, r=0.05)
    assert "delta" in g
    assert "gamma" in g
    assert "theta" in g
    assert "vega" in g
    assert 0 <= g["delta"] <= 1
    assert g["gamma"] >= 0
    assert g["vega"] >= 0


def test_compute_greeks_put():
    pytest.importorskip("py_vollib")
    g = compute_greeks(flag="p", S=100.0, K=100.0, t=0.25, sigma=0.25)
    assert -1 <= g["delta"] <= 0


def test_compute_greeks_invalid_returns_empty():
    assert compute_greeks(flag="c", S=100.0, K=100.0, t=0.0, sigma=0.25) == {}
    assert compute_greeks(flag="c", S=100.0, K=100.0, t=0.25, sigma=0.0) == {}


def test_polygon_normalize_row():
    """Polygon snapshot result -> OptionsSnapshot with optional Greeks fill-in."""
    provider = PolygonOptionsProvider(api_key="test-key")
    r = {
        "details": {
            "ticker": "O:AAPL250117C00150000",
            "expiration_date": "2025-01-17",
            "strike_price": 150.0,
            "contract_type": "call",
        },
        "day": {"close": 5.25, "volume": 100},
        "last_quote": {"bid": 5.2, "ask": 5.3},
        "implied_volatility": 0.25,
        "greeks": {"delta": 0.55, "gamma": 0.02, "theta": -0.05, "vega": 0.11},
        "open_interest": 500,
        "underlying_asset": {"price": 148.0},
    }
    row = provider._normalize_row("AAPL", date(2025, 1, 2), r)
    assert row is not None
    assert row.underlying == "AAPL"
    assert row.contract_id == "O:AAPL250117C00150000"
    assert row.expiry == date(2025, 1, 17)
    assert row.strike == 150.0
    assert row.option_type == "call"
    assert row.bid == 5.2
    assert row.ask == 5.3
    assert row.mid == 5.25
    assert row.close == 5.25
    assert row.iv == 0.25
    assert row.delta == 0.55
    assert row.source == "polygon"
    assert row.volume == 100
    assert row.open_interest == 500


def test_polygon_normalize_row_missing_greeks_computed():
    """When greeks missing but IV and underlying price present, compute via py_vollib."""
    pytest.importorskip("py_vollib")
    provider = PolygonOptionsProvider(api_key="test-key")
    r = {
        "details": {
            "ticker": "O:SPY250117C00600000",
            "expiration_date": "2025-01-17",
            "strike_price": 600.0,
            "contract_type": "call",
        },
        "day": {"close": 2.5},
        "implied_volatility": 0.20,
        "greeks": {},
        "underlying_asset": {"price": 598.0},
    }
    row = provider._normalize_row("SPY", date(2025, 1, 2), r)
    assert row is not None
    assert row.delta is not None
    assert row.gamma is not None
    assert row.theta is not None
    assert row.vega is not None


def test_polygon_fetch_chain_snapshot_paginated():
    """fetch_chain_snapshot uses next_url when present."""
    provider = PolygonOptionsProvider(api_key="test-key")
    data1 = {
        "results": [
            {
                "details": {"ticker": "O:AAPL250117C00150000", "expiration_date": "2025-01-17", "strike_price": 150, "contract_type": "call"},
                "day": {},
                "implied_volatility": 0.25,
                "greeks": {"delta": 0.5},
                "underlying_asset": {"price": 148},
            }
        ],
        "next_url": "https://api.polygon.io/v3/snapshot/options/AAPL?cursor=next",
    }
    data2 = {"results": []}
    with patch.object(provider, "_get", side_effect=[data1, data2]):
        rows = provider.fetch_chain_snapshot("AAPL", snapshot_date=date(2025, 1, 2))
    assert len(rows) == 1
    assert rows[0].contract_id == "O:AAPL250117C00150000"


def test_tradier_normalize_row():
    """Tradier chain option -> OptionsSnapshot."""
    provider = TradierOptionsProvider(api_token="test-token")
    opt = {
        "symbol": "AAPL210416C00125000",
        "strike": 125,
        "option_type": "call",
        "bid": 3.4,
        "ask": 3.45,
        "close": 3.15,
        "volume": 1105,
        "open_interest": 8249,
        "expiration_date": "2021-04-16",
        "greeks": {"delta": 0.6, "gamma": 0.03, "theta": -0.1, "vega": 0.15, "smv_vol": 0.22},
    }
    row = provider._normalize_row("AAPL", date(2021, 4, 15), date(2021, 4, 16), opt)
    assert row is not None
    assert row.underlying == "AAPL"
    assert row.contract_id == "AAPL210416C00125000"
    assert row.strike == 125
    assert row.option_type == "call"
    assert row.iv == 0.22
    assert row.delta == 0.6
    assert row.source == "tradier"


def test_tradier_fetch_chain_structure():
    """Tradier chains response options.option array."""
    provider = TradierOptionsProvider(api_token="test-token")
    data = {
        "options": {
            "option": [
                {
                    "symbol": "SPY250117C00600000",
                    "strike": 600,
                    "option_type": "call",
                    "bid": 1.0,
                    "ask": 1.05,
                    "greeks": {"delta": 0.5, "smv_vol": 0.18},
                }
            ]
        }
    }
    with patch.object(provider, "_get", return_value=data):
        rows = provider.fetch_chain("SPY", date(2025, 1, 17), snapshot_date=date(2025, 1, 2))
    assert len(rows) == 1
    assert rows[0].contract_id == "SPY250117C00600000"


def test_get_polygon_options_provider():
    with patch.dict("os.environ", {"MASSIVE_API": "key"}, clear=False):
        p = get_polygon_options_provider()
    assert isinstance(p, PolygonOptionsProvider)
    assert p.api_key == "key"


def test_get_tradier_options_provider():
    with patch.dict("os.environ", {"TRADIER_API_TOKEN": "token"}, clear=False):
        p = get_tradier_options_provider(sandbox=True)
    assert isinstance(p, TradierOptionsProvider)
    assert p.api_token == "token"
    assert "sandbox" in p.base_url


def test_write_options_snapshots_calls_cursor():
    mock_conn = MagicMock()
    rows = [
        OptionsSnapshot(
            underlying="SPY",
            snapshot_date=date(2025, 1, 2),
            contract_id="O:SPY250117C00600000",
            expiry=date(2025, 1, 17),
            strike=600.0,
            option_type="call",
            bid=1.0,
            ask=1.05,
            mid=1.025,
            iv=0.20,
            delta=0.5,
            source="polygon",
        ),
    ]
    write_options_snapshots(mock_conn, rows)
    assert mock_conn.cursor.return_value.__enter__.return_value.execute.call_count == 1
    call_args = mock_conn.cursor.return_value.__enter__.return_value.execute.call_args[0]
    assert "options_snapshots" in call_args[0]
    assert call_args[1][0] == "SPY"
    assert call_args[1][2] == "O:SPY250117C00600000"


# ==================== YFinanceOptionsProvider tests ====================


def _make_chain_df(strikes, iv=0.25, bid=5.0, ask=5.2, volume=100, oi=500, prefix="SPY260120"):
    """Build a fake calls/puts DataFrame matching yfinance column layout."""
    rows = []
    for s in strikes:
        rows.append({
            "contractSymbol": f"{prefix}C{int(s*1000):08d}",
            "strike": float(s),
            "lastPrice": (bid + ask) / 2.0,
            "bid": bid,
            "ask": ask,
            "impliedVolatility": iv,
            "volume": volume,
            "openInterest": oi,
            "change": 0.0,
            "percentChange": 0.0,
            "inTheMoney": False,
            "lastTradeDate": pd.Timestamp("2026-01-15 15:00:00+00:00"),
            "contractSize": "REGULAR",
            "currency": "USD",
        })
    return pd.DataFrame(rows)


class FakeOptionChain:
    """Mimics the namedtuple returned by yfinance Ticker.option_chain()."""
    def __init__(self, calls_df, puts_df):
        self.calls = calls_df
        self.puts = puts_df


class FakeFastInfo:
    last_price = 600.0
    previous_close = 598.0


class FakeTicker:
    """Minimal yfinance Ticker stub."""
    options = ("2026-01-20", "2026-02-03", "2026-02-17")
    fast_info = FakeFastInfo()

    def __init__(self, strikes=None):
        self._strikes = strikes or [580, 590, 600, 610, 620]

    def option_chain(self, exp_str):
        calls = _make_chain_df(self._strikes)
        puts = _make_chain_df(self._strikes)
        return FakeOptionChain(calls, puts)

    def history(self, period="3mo", interval="1d"):
        """Return a DataFrame with 60 fake daily closes."""
        closes = np.linspace(590, 600, 60)
        idx = pd.date_range("2025-11-01", periods=60, freq="B")
        return pd.DataFrame({"Close": closes}, index=idx)


def test_yfinance_provider_basic():
    """YFinanceOptionsProvider returns OptionsSnapshots with correct fields."""
    provider = YFinanceOptionsProvider(expiry_targets_days=[7, 14, 30])
    with patch("yfinance.Ticker", return_value=FakeTicker()):
        rows = provider.fetch_chain_snapshot("SPY", snapshot_date=date(2026, 1, 13))

    assert len(rows) > 0
    for row in rows:
        assert isinstance(row, OptionsSnapshot)
        assert row.underlying == "SPY"
        assert row.source == "yfinance"
        assert row.strike > 0
        assert row.option_type in ("call", "put")
        assert row.iv is not None and row.iv > 0
        assert row.bid is not None
        assert row.ask is not None
        assert row.mid is not None


def test_yfinance_provider_computes_greeks():
    """Greeks are computed from IV via compute_greeks (py_vollib)."""
    pytest.importorskip("py_vollib")
    provider = YFinanceOptionsProvider(expiry_targets_days=[14])
    with patch("yfinance.Ticker", return_value=FakeTicker()):
        rows = provider.fetch_chain_snapshot("SPY", snapshot_date=date(2026, 1, 13))

    calls = [r for r in rows if r.option_type == "call"]
    assert len(calls) > 0
    for row in calls:
        assert row.delta is not None, "delta should be computed"
        assert row.gamma is not None, "gamma should be computed"
        assert row.theta is not None, "theta should be computed"
        assert row.vega is not None, "vega should be computed"


def test_yfinance_provider_filters_expirations():
    """Only the expirations closest to the target DTEs are fetched."""
    provider = YFinanceOptionsProvider(expiry_targets_days=[7])
    with patch("yfinance.Ticker", return_value=FakeTicker()):
        rows = provider.fetch_chain_snapshot("SPY", snapshot_date=date(2026, 1, 13))

    # Only 1 expiration target => all rows should share the same expiry
    expiries = {r.expiry for r in rows}
    assert len(expiries) == 1
    assert date(2026, 1, 20) in expiries  # closest to 7 days from Jan 13


def test_yfinance_provider_filters_strikes():
    """Strikes are filtered to ATM ± 1σ / 2σ (5 targets)."""
    wide_strikes = list(range(400, 800, 10))  # 40 strikes
    provider = YFinanceOptionsProvider(expiry_targets_days=[14])
    with patch("yfinance.Ticker", return_value=FakeTicker(strikes=wide_strikes)):
        rows = provider.fetch_chain_snapshot("SPY", snapshot_date=date(2026, 1, 13))

    strikes_per_type = {}
    for r in rows:
        key = (r.expiry, r.option_type)
        strikes_per_type.setdefault(key, set()).add(r.strike)

    for key, strikes in strikes_per_type.items():
        assert len(strikes) <= 5, f"Expected ≤5 strikes per expiry/type, got {len(strikes)} for {key}"


def test_yfinance_select_expirations():
    provider = YFinanceOptionsProvider(expiry_targets_days=[7, 14, 30])
    avail = [date(2026, 1, 20), date(2026, 1, 27), date(2026, 2, 10), date(2026, 3, 20)]
    selected = provider._select_expirations(date(2026, 1, 13), avail)
    assert date(2026, 1, 20) in selected  # ~7 DTE
    assert date(2026, 1, 27) in selected  # ~14 DTE
    assert date(2026, 2, 10) in selected  # ~28 DTE, closest to 30
    assert len(selected) == 3


def test_yfinance_select_strikes():
    available = np.array([580, 585, 590, 595, 600, 605, 610, 615, 620])
    targets = [590.0, 595.0, 600.0, 605.0, 610.0]
    selected = YFinanceOptionsProvider._select_strikes(available, targets)
    assert 600.0 in selected  # ATM
    assert len(selected) == 5


def test_yfinance_spot_price_fast_info():
    provider = YFinanceOptionsProvider()
    price = provider._spot_price(FakeTicker())
    assert price == 600.0


def test_yfinance_spot_price_fallback_to_info():
    """Falls back to ticker.info when fast_info fails."""

    class NoFastInfo:
        @property
        def fast_info(self):
            raise AttributeError("no fast_info")
        info = {"regularMarketPrice": 550.0}
        options = ()

    price = YFinanceOptionsProvider._spot_price(NoFastInfo())
    assert price == 550.0


def test_yfinance_no_expirations_returns_empty():
    class EmptyTicker:
        fast_info = FakeFastInfo()
        options = ()

    provider = YFinanceOptionsProvider()
    with patch("yfinance.Ticker", return_value=EmptyTicker()):
        rows = provider.fetch_chain_snapshot("SPY", snapshot_date=date(2026, 1, 13))
    assert rows == []


def test_yfinance_estimate_hist_vol():
    provider = YFinanceOptionsProvider()
    vol = provider._estimate_hist_vol(FakeTicker())
    assert 0 < vol < 1.0  # should be a reasonable number


def test_safe_float_nan():
    """_safe_float returns None for NaN values."""
    row = {"val": float("nan"), "good": 1.5, "missing": None}
    assert _safe_float(row, "val") is None
    assert _safe_float(row, "good") == 1.5
    assert _safe_float(row, "missing") is None
    assert _safe_float(row, "nonexistent") is None


def test_get_yfinance_options_provider():
    p = get_yfinance_options_provider(expiry_targets_days=[7, 30])
    assert isinstance(p, YFinanceOptionsProvider)
    assert p.expiry_targets == [7, 30]


# ==================== Backfill fallback tests ====================


def test_backfill_polygon_403_falls_back_to_yfinance():
    """When Polygon returns 403, backfill transparently falls back to yfinance."""
    import httpx

    from src.ingestion.backfill_options import _try_polygon

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_req = MagicMock()
    exc = httpx.HTTPStatusError("Forbidden", request=mock_req, response=mock_resp)

    polygon = MagicMock(spec=PolygonOptionsProvider)
    polygon.fetch_chain_snapshot.side_effect = exc

    rows, is_auth_err = _try_polygon(polygon, "SPY", date(2026, 1, 13))
    assert rows == []
    assert is_auth_err is True


def test_backfill_polygon_success_no_fallback():
    """When Polygon succeeds, yfinance is not invoked."""
    from src.ingestion.backfill_options import _try_polygon

    fake_row = OptionsSnapshot(
        underlying="SPY", snapshot_date=date(2026, 1, 13),
        contract_id="O:SPY", expiry=date(2026, 1, 20),
        strike=600, option_type="call", source="polygon",
    )
    polygon = MagicMock(spec=PolygonOptionsProvider)
    polygon.fetch_chain_snapshot.return_value = [fake_row]

    rows, is_auth_err = _try_polygon(polygon, "SPY", date(2026, 1, 13))
    assert len(rows) == 1
    assert is_auth_err is False


def test_backfill_polygon_other_error_not_auth():
    """Non-auth errors from Polygon do not trigger yfinance fallback."""
    import httpx

    from src.ingestion.backfill_options import _try_polygon

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_req = MagicMock()
    exc = httpx.HTTPStatusError("Server Error", request=mock_req, response=mock_resp)

    polygon = MagicMock(spec=PolygonOptionsProvider)
    polygon.fetch_chain_snapshot.side_effect = exc

    rows, is_auth_err = _try_polygon(polygon, "SPY", date(2026, 1, 13))
    assert rows == []
    assert is_auth_err is False


def test_backfill_run_uses_yfinance_after_403():
    """Full run_backfill_options switches to yfinance after first 403."""
    import httpx

    from src.ingestion.backfill_options import run_backfill_options

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_req = MagicMock()
    polygon_exc = httpx.HTTPStatusError("Forbidden", request=mock_req, response=mock_resp)

    yf_rows = [
        OptionsSnapshot(
            underlying="SPY", snapshot_date=date(2026, 1, 13),
            contract_id="SPY260120C00600000", expiry=date(2026, 1, 20),
            strike=600, option_type="call", source="yfinance",
        ),
    ]

    with (
        patch("src.ingestion.backfill_options.apply_migrations"),
        patch("src.ingestion.backfill_options.load_universe", return_value=["SPY"]),
        patch("src.ingestion.backfill_options.get_polygon_options_provider") as mock_poly_factory,
        patch("src.ingestion.backfill_options.get_yfinance_options_provider") as mock_yf_factory,
        patch("src.ingestion.backfill_options.get_connection"),
        patch("src.ingestion.backfill_options.write_options_snapshots") as mock_write,
    ):
        mock_poly = MagicMock()
        mock_poly.fetch_chain_snapshot.side_effect = polygon_exc
        mock_poly_factory.return_value = mock_poly

        mock_yf = MagicMock()
        mock_yf.fetch_chain_snapshot.return_value = yf_rows
        mock_yf_factory.return_value = mock_yf

        run_backfill_options(symbols=["SPY"], snapshot_date=date(2026, 1, 13))

    # yfinance was used and its rows were written
    mock_yf.fetch_chain_snapshot.assert_called_once_with("SPY", snapshot_date=date(2026, 1, 13))
    mock_write.assert_called_once()
    written = mock_write.call_args[0][1]
    assert len(written) == 1
    assert written[0].source == "yfinance"
