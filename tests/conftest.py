"""Pytest fixtures and shared test config."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src is on path when running tests from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

# Optional: allow tests to run without py_clob_client installed (stub module)
try:
    import py_clob_client  # noqa: F401
except ImportError:
    sys.modules["py_clob_client"] = MagicMock()
    sys.modules["py_clob_client.client"] = MagicMock()
    sys.modules["py_clob_client.client"].ClobClient = MagicMock


@pytest.fixture
def sample_gamma_event():
    """Minimal Gamma API event payload."""
    return {
        "id": "0xabc",
        "title": "Fed rate decision March 2026",
        "slug": "fed-rate-march-2026",
        "category": "Economics",
        "startDate": "2026-03-18T14:00:00.000Z",
        "endDate": "2026-03-18T18:00:00.000Z",
        "closed": True,
        "closedTime": "2026-03-19T00:00:00.000Z",
        "markets": [
            {
                "id": "0xmarket1",
                "question": "Will Fed raise rates?",
                "slug": "fed-raise-march",
                "outcomes": "Yes,No",
                "outcomePrices": "0.35,0.65",
                "clobTokenIds": ["0xtoken1", "0xtoken2"],
                "volumeNum": 100000.0,
                "liquidityNum": 50000.0,
                "active": False,
            }
        ],
    }


@pytest.fixture
def sample_kalshi_event():
    """Minimal Kalshi API event with nested markets."""
    return {
        "event_ticker": "FED-26MAR",
        "title": "Federal Reserve rate decision",
        "sub_title": "Fed March 2026",
        "category": "Economics",
        "status": "closed",
        "close_time": "2026-03-18T18:00:00Z",
        "markets": [
            {
                "ticker": "FED-26MAR-T4.50",
                "title": "Rate at least 4.50%",
                "status": "closed",
                "last_price_dollars": "0.72",
                "volume": 5000,
                "liquidity_dollars": "10000.00",
            }
        ],
    }


@pytest.fixture
def sample_clob_prices_history():
    """Minimal CLOB /prices-history response."""
    return [
        {"t": 1700000000000, "p": 0.35},
        {"t": 1700003600000, "p": 0.38},
        {"t": 1700007200000, "p": 0.40},
    ]
