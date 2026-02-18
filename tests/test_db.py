"""Tests for DB migration and writer (with mocked connection)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.db import apply_migrations, get_connection_string, run_migration
from src.ingestion.pm_writer import write_pm_events, write_pm_markets, write_pm_prices
from src.utils.schemas import PMEvent, PMMarket, PMPrice


def test_get_connection_string_from_env():
    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://u:p@h:5432/d"}, clear=False):
        assert "postgresql://u:p@h:5432/d" in get_connection_string()
    with patch.dict("os.environ", {}, clear=True):
        with patch.dict("os.environ", {"POSTGRES_HOST": "db", "POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "mydb"}):
            url = get_connection_string()
            assert "db" in url and "u" in url and "mydb" in url


def test_run_migration_executes_sql():
    mock_conn = MagicMock()
    run_migration(mock_conn, "SELECT 1")
    mock_conn.cursor.return_value.__enter__.return_value.execute.assert_called_once_with("SELECT 1")


def test_write_pm_events_calls_cursor():
    mock_conn = MagicMock()
    events = [
        PMEvent(event_id="e1", platform="polymarket", title="E1"),
    ]
    write_pm_events(mock_conn, events)
    assert mock_conn.cursor.return_value.__enter__.return_value.execute.call_count == 1


def test_write_pm_markets_calls_cursor():
    mock_conn = MagicMock()
    markets = [
        PMMarket(market_id="m1", event_id="e1", platform="polymarket", slug="s1", token_ids=["t1"]),
    ]
    write_pm_markets(mock_conn, markets)
    assert mock_conn.cursor.return_value.__enter__.return_value.execute.call_count == 1


def test_write_pm_prices_calls_cursor():
    mock_conn = MagicMock()
    ts = datetime.now(timezone.utc)
    prices = [
        PMPrice(token_id="t1", platform="polymarket", ts=ts, price=0.5),
    ]
    write_pm_prices(mock_conn, prices)
    assert mock_conn.cursor.return_value.__enter__.return_value.execute.call_count == 1
