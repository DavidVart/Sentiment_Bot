"""Tests for Kalshi API client."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.connectors.kalshi import KalshiClient
from src.utils.schemas import PMEvent, PMMarket, PMPrice


def test_kalshi_normalize_event(sample_kalshi_event):
    client = KalshiClient()
    pe = client.normalize_event(sample_kalshi_event)
    assert isinstance(pe, PMEvent)
    assert pe.platform == "kalshi"
    assert pe.event_id == "kalshi_FED-26MAR"
    assert "Federal" in pe.title
    assert pe.status == "closed"


def test_kalshi_normalize_markets_for_event(sample_kalshi_event):
    client = KalshiClient()
    event_id = "kalshi_FED-26MAR"
    markets = client.normalize_markets_for_event(event_id, sample_kalshi_event)
    assert len(markets) == 1
    m = markets[0]
    assert isinstance(m, PMMarket)
    assert m.platform == "kalshi"
    assert m.event_id == event_id
    assert m.token_ids == ["FED-26MAR-T4.50"]
    assert m.slug == "FED-26MAR-T4.50"
    assert m.active is False
    assert m.volume == 5000
    assert m.liquidity == 10000.0


def test_kalshi_candlestick_parsing():
    """Test get_market_candlesticks parses response into PMPrice list."""
    client = KalshiClient()
    raw = {
        "candlesticks": [
            {"end_ts": 1700000000, "yes_price": 0.35},
            {"end_ts": 1700003600, "yes_price": 0.38},
        ]
    }

    with patch.object(client, "_get", return_value=raw):
        prices = client.get_market_candlesticks(
            "FED-26MAR-T4.50",
            start_ts=__import__("datetime").datetime.fromtimestamp(1699900000, tz=__import__("datetime").timezone.utc),
            end_ts=__import__("datetime").datetime.fromtimestamp(1700100000, tz=__import__("datetime").timezone.utc),
        )
    assert len(prices) == 2
    for p in prices:
        assert isinstance(p, PMPrice)
        assert p.token_id == "FED-26MAR-T4.50"
        assert p.platform == "kalshi"
        assert 0 <= p.price <= 1
    assert prices[0].price == 0.35
    assert prices[1].price == 0.38
