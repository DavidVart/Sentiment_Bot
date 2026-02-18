"""Tests for Polymarket Gamma and CLOB connectors."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.connectors.polymarket import ClobClient, GammaClient
from src.utils.schemas import PMEvent, PMMarket, PMPrice


def test_gamma_normalize_event(sample_gamma_event):
    client = GammaClient()
    pe = client.normalize_event(sample_gamma_event)
    assert isinstance(pe, PMEvent)
    assert pe.platform == "polymarket"
    assert pe.event_id == "polymarket_0xabc"
    assert "Fed" in pe.title
    assert pe.status == "closed"
    assert pe.start_ts is not None
    assert pe.end_ts is not None


def test_gamma_normalize_markets_for_event(sample_gamma_event):
    client = GammaClient()
    event_id = "polymarket_0xabc"
    markets = client.normalize_markets_for_event(event_id, sample_gamma_event)
    assert len(markets) == 1
    m = markets[0]
    assert isinstance(m, PMMarket)
    assert m.platform == "polymarket"
    assert m.event_id == event_id
    assert m.token_ids == ["0xtoken1", "0xtoken2"]
    assert m.volume == 100000.0
    assert m.liquidity == 50000.0
    assert m.active is False


def test_clob_parse_prices_history(sample_clob_prices_history):
    """Test that get_prices_history parses response into PMPrice list."""
    client = ClobClient()

    def mock_get(url, params=None, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return sample_clob_prices_history

        return Resp()

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get = mock_get
        prices = client.get_prices_history("0xtoken1", interval="1d")
    assert len(prices) == 3
    for p in prices:
        assert isinstance(p, PMPrice)
        assert p.token_id == "0xtoken1"
        assert p.platform == "polymarket"
        assert 0 <= p.price <= 1
        assert p.source == "rest"
    assert prices[0].ts == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert prices[0].price == 0.35


def test_gamma_token_ids_from_string():
    """Token IDs can be comma-separated string (via normalize_markets)."""
    client = GammaClient()
    event = {
        "id": "e1",
        "title": "E",
        "markets": [{"id": "m1", "clobTokenIds": "id1,id2", "outcomes": "Y,N"}],
    }
    markets = client.normalize_markets_for_event("polymarket_e1", event)
    assert len(markets) == 1
    assert markets[0].token_ids == ["id1", "id2"]
