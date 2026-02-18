"""Shared connector tests (retry behavior, config loading)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.connectors.polymarket import GammaClient
from src.utils.http_utils import with_retry


def test_with_retry_succeeds_first():
    fn = MagicMock(return_value=42)
    assert with_retry(fn, max_retries=3) == 42
    assert fn.call_count == 1


def test_with_retry_succeeds_after_failures():
    fn = MagicMock(side_effect=[httpx.HTTPError("err"), httpx.HTTPError("err"), 43])
    assert with_retry(fn, max_retries=5, backoff_base=0.01) == 43
    assert fn.call_count == 3


def test_with_retry_raises_after_max():
    fn = MagicMock(side_effect=httpx.HTTPError("err"))
    with pytest.raises(httpx.HTTPError):
        with_retry(fn, max_retries=2, backoff_base=0.01)
    assert fn.call_count == 2


def test_gamma_client_get_events_structure():
    """Gamma get_events returns list (mocked)."""
    client = GammaClient()
    with patch.object(client, "_get", return_value=[{"id": "e1", "title": "Event 1"}]):
        events = client.get_events(limit=10)
    assert isinstance(events, list)
    assert len(events) == 1
    assert events[0]["id"] == "e1"
