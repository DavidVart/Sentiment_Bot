"""Tests for pm_feature_builder (derived features from price series)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.features.pm_feature_builder import _logit, _build_for_token
from src.utils.schemas import PMFeature


def test_logit():
    assert _logit(0.5) == 0.0
    assert _logit(0.25) < 0
    assert _logit(0.75) > 0
    assert abs(_logit(0.5) - 0.0) < 1e-9
    # edge
    assert _logit(0.0) == 0.0
    assert _logit(1.0) == 0.0


def test_build_for_token_no_prices():
    """With no rows, _build_for_token should not write."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = []
    mock_cursor_cm = MagicMock()
    mock_cursor_cm.__enter__.return_value = mock_cur
    mock_cursor_cm.__exit__.return_value = False
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor_cm
    mock_conn_cm = MagicMock()
    mock_conn_cm.__enter__.return_value = mock_conn
    mock_conn_cm.__exit__.return_value = False
    with patch("src.features.pm_feature_builder.get_connection", return_value=mock_conn_cm):
        with patch("src.features.pm_feature_builder.write_pm_features") as mock_write:
            _build_for_token("test_token", 1)
    assert not mock_write.called


def test_build_for_token_with_prices():
    """With a few price rows, features are computed and write_pm_features called."""
    from datetime import timedelta

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        (base, 0.30),
        (base + timedelta(hours=1), 0.32),
        (base + timedelta(hours=2), 0.35),
        (base + timedelta(hours=3), 0.34),
        (base + timedelta(hours=4), 0.36),
    ]
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_cursor_cm = MagicMock()
    mock_cursor_cm.__enter__.return_value = mock_cur
    mock_cursor_cm.__exit__.return_value = False
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor_cm
    mock_conn_cm = MagicMock()
    mock_conn_cm.__enter__.return_value = mock_conn
    mock_conn_cm.__exit__.return_value = False
    with patch("src.features.pm_feature_builder.get_connection", return_value=mock_conn_cm):
        with patch("src.features.pm_feature_builder.write_pm_features") as mock_write:
            _build_for_token("test_token", 1)
    assert mock_write.called
    call_args = mock_write.call_args
    features = call_args[0][1]
    assert len(features) == len(rows)
    for f in features:
        assert isinstance(f, PMFeature)
        assert f.token_id == "test_token"
        assert 0 <= f.p <= 1
        assert f.logit_p == _logit(f.p)
    assert features[0].p == 0.30
    assert features[1].delta_p_1h is not None  # 0.32 - 0.30
