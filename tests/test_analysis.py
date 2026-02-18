"""Tests for analysis module: table formatting and plot generation with mock data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.agents.analysis import (
    format_ablation_table,
    format_regime_table,
    load_ablation_results,
    plot_drawdown_over_time,
    plot_equity_curves,
    plot_exposure_over_time,
    plot_rolling_sharpe,
    generate_event_case_study,
)


def _mock_ablation_data():
    return {
        "results": [
            {"variant": "A", "algorithm": "ppo", "seed": s, "sharpe": 0.5, "sortino": 0.6, "calmar": 0.4, "max_drawdown": 0.05, "hit_rate": 52.0, "turnover": 1.0, "regime_metrics_test": {"low": {"annualized_sharpe": 0.3, "sortino": 0.4, "calmar": 0.2, "max_drawdown": 0.03, "hit_rate_pct": 50.0, "turnover_rate": 0.8, "n_bars": 100}, "medium": {"annualized_sharpe": 0.5, "sortino": 0.6, "calmar": 0.4, "max_drawdown": 0.05, "hit_rate_pct": 52.0, "turnover_rate": 1.0, "n_bars": 80}, "high": {"annualized_sharpe": 0.4, "sortino": 0.5, "calmar": 0.3, "max_drawdown": 0.06, "hit_rate_pct": 48.0, "turnover_rate": 1.2, "n_bars": 20}}}
            for s in range(5)
        ]
        + [
            {"variant": "B", "algorithm": "ppo", "seed": s, "sharpe": 0.6, "sortino": 0.7, "calmar": 0.5, "max_drawdown": 0.04, "hit_rate": 55.0, "turnover": 1.1, "regime_metrics_test": {"low": {"annualized_sharpe": 0.4, "sortino": 0.5, "calmar": 0.3, "max_drawdown": 0.03, "hit_rate_pct": 52.0, "turnover_rate": 0.9, "n_bars": 100}, "medium": {"annualized_sharpe": 0.6, "sortino": 0.7, "calmar": 0.5, "max_drawdown": 0.04, "hit_rate_pct": 55.0, "turnover_rate": 1.1, "n_bars": 80}, "high": {"annualized_sharpe": 0.5, "sortino": 0.6, "calmar": 0.4, "max_drawdown": 0.05, "hit_rate_pct": 52.0, "turnover_rate": 1.2, "n_bars": 20}}}
            for s in range(5)
        ]
        + [
            {"variant": "C", "algorithm": "ppo", "seed": s, "sharpe": 0.55, "sortino": 0.65, "calmar": 0.45, "max_drawdown": 0.045, "hit_rate": 53.0, "turnover": 1.05, "regime_metrics_test": {"low": {"annualized_sharpe": 0.35, "sortino": 0.45, "calmar": 0.25, "max_drawdown": 0.03, "hit_rate_pct": 51.0, "turnover_rate": 0.85, "n_bars": 100}, "medium": {"annualized_sharpe": 0.55, "sortino": 0.65, "calmar": 0.45, "max_drawdown": 0.045, "hit_rate_pct": 53.0, "turnover_rate": 1.05, "n_bars": 80}, "high": {"annualized_sharpe": 0.45, "sortino": 0.55, "calmar": 0.35, "max_drawdown": 0.055, "hit_rate_pct": 50.0, "turnover_rate": 1.15, "n_bars": 20}}}
            for s in range(5)
        ]
        + [
            {"variant": "D", "algorithm": "ppo", "seed": s, "sharpe": 0.65, "sortino": 0.75, "calmar": 0.55, "max_drawdown": 0.04, "hit_rate": 56.0, "turnover": 1.15, "regime_metrics_test": {"low": {"annualized_sharpe": 0.45, "sortino": 0.55, "calmar": 0.35, "max_drawdown": 0.03, "hit_rate_pct": 54.0, "turnover_rate": 1.0, "n_bars": 100}, "medium": {"annualized_sharpe": 0.65, "sortino": 0.75, "calmar": 0.55, "max_drawdown": 0.04, "hit_rate_pct": 56.0, "turnover_rate": 1.15, "n_bars": 80}, "high": {"annualized_sharpe": 0.55, "sortino": 0.65, "calmar": 0.45, "max_drawdown": 0.05, "hit_rate_pct": 54.0, "turnover_rate": 1.2, "n_bars": 20}}}
            for s in range(5)
        ],
        "aggregated": [
            {"variant": "A", "algorithm": "ppo", "sharpe_mean": 0.5, "sharpe_std": 0.05, "sortino_mean": 0.6, "sortino_std": 0.06, "calmar_mean": 0.4, "calmar_std": 0.04, "max_drawdown_mean": 0.05, "max_drawdown_std": 0.005, "hit_rate_mean": 52.0, "hit_rate_std": 2.0, "turnover_mean": 1.0, "turnover_std": 0.1},
            {"variant": "B", "algorithm": "ppo", "sharpe_mean": 0.6, "sharpe_std": 0.06, "sortino_mean": 0.7, "sortino_std": 0.07, "calmar_mean": 0.5, "calmar_std": 0.05, "max_drawdown_mean": 0.04, "max_drawdown_std": 0.004, "hit_rate_mean": 55.0, "hit_rate_std": 2.2, "turnover_mean": 1.1, "turnover_std": 0.11},
            {"variant": "C", "algorithm": "ppo", "sharpe_mean": 0.55, "sharpe_std": 0.055, "sortino_mean": 0.65, "sortino_std": 0.065, "calmar_mean": 0.45, "calmar_std": 0.045, "max_drawdown_mean": 0.045, "max_drawdown_std": 0.0045, "hit_rate_mean": 53.0, "hit_rate_std": 2.1, "turnover_mean": 1.05, "turnover_std": 0.105},
            {"variant": "D", "algorithm": "ppo", "sharpe_mean": 0.65, "sharpe_std": 0.065, "sortino_mean": 0.75, "sortino_std": 0.075, "calmar_mean": 0.55, "calmar_std": 0.055, "max_drawdown_mean": 0.04, "max_drawdown_std": 0.004, "hit_rate_mean": 56.0, "hit_rate_std": 2.2, "turnover_mean": 1.15, "turnover_std": 0.115},
        ],
        "pvalues_vs_A": {
            "pval_sharpe_vs_A_B_ppo": 0.02,
            "pval_sharpe_vs_A_C_ppo": 0.05,
            "pval_sharpe_vs_A_D_ppo": 0.01,
        },
        "config": {},
    }


def test_load_ablation_results(tmp_path):
    path = tmp_path / "ablation.json"
    data = _mock_ablation_data()
    with open(path, "w") as f:
        json.dump(data, f)
    loaded = load_ablation_results(path)
    assert loaded["aggregated"][0]["variant"] == "A"
    assert len(loaded["results"]) == 20


def test_format_ablation_table():
    data = _mock_ablation_data()
    csv_content, md_content = format_ablation_table(data, algorithm="ppo")
    assert "Base" in csv_content and "Sharpe" in csv_content
    assert "±" in csv_content or "std" in md_content or "0.5000" in csv_content
    assert "|" in md_content
    lines = csv_content.strip().split("\n")
    assert len(lines) >= 5


def test_format_ablation_table_empty():
    csv_content, md_content = format_ablation_table({"aggregated": []}, algorithm="ppo")
    assert csv_content == ""
    assert md_content == ""


def test_format_regime_table():
    data = _mock_ablation_data()
    csv_content, md_content = format_regime_table(data, algorithm="ppo")
    assert "low" in csv_content and "medium" in csv_content and "high" in csv_content
    assert "Variant" in csv_content


def test_format_regime_table_empty():
    csv_content, md_content = format_regime_table({"results": []}, algorithm="ppo")
    assert csv_content == ""
    assert md_content == ""


def _has_mpl():
    from src.agents import analysis as am
    return getattr(am, "_HAS_MPL", False)


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
def test_plot_equity_curves(tmp_path):
    series = {
        "A": np.cumsum(np.random.randn(50)) + 100,
        "B": np.cumsum(np.random.randn(50)) + 100,
    }
    out = tmp_path / "equity.png"
    plot_equity_curves(series, out, title="Test")
    assert out.exists()


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
def test_plot_equity_curves_with_vix(tmp_path):
    series = {"A": np.ones(30) * 100}
    vix = np.array([10.0] * 10 + [20.0] * 10 + [30.0] * 10)
    out = tmp_path / "equity_vix.png"
    plot_equity_curves(series, out, vix_series=vix)
    assert out.exists()


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
def test_plot_drawdown_over_time(tmp_path):
    series = {"A": 100 + np.cumsum(np.random.randn(40)), "B": 100 + np.cumsum(np.random.randn(40))}
    out = tmp_path / "dd.png"
    plot_drawdown_over_time(series, out)
    assert out.exists()


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
def test_plot_rolling_sharpe(tmp_path):
    n = 50
    pnl = np.random.randn(n) * 10
    eq = 100_000 + np.cumsum(pnl)
    series = {"A": (pnl, eq)}
    out = tmp_path / "rolling_sharpe.png"
    plot_rolling_sharpe(series, out, window=30)
    assert out.exists()


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
def test_plot_exposure_over_time(tmp_path):
    delta = np.random.randn(40) * 5
    vega = np.random.randn(40) * 10
    out = tmp_path / "exposure.png"
    plot_exposure_over_time(delta, vega, out)
    assert out.exists()


@pytest.mark.skipif(not _has_mpl(), reason="matplotlib not installed")
def test_generate_event_case_study_mock_policy(tmp_path):
    from src.agents.baselines import BuyAndHold
    bars = [
        {
            "underlying": "SPY",
            "equity_return_1d": 0.001,
            "realized_vol_5d": 0.15, "realized_vol_10d": 0.16, "realized_vol_20d": 0.17, "realized_vol_60d": 0.18,
            "vix_close": 18.0, "iv_term_slope": -0.01, "iv_skew": 0.02, "options_gap_flag": False,
            "atm_iv_7d": 0.20, "atm_iv_14d": 0.19, "atm_iv_30d": 0.18 + 0.01 * (i % 5),
            "pm_p": 0.4 + 0.1 * (i % 6),
            "sent_news_asset": 0.1, "sent_social_asset": 0.0, "sent_macro_topic": -0.05,
            "sent_dispersion": 0.02, "sent_momentum": 0.01, "sent_volume": 5, "no_news_flag": False,
        }
        for i in range(25)
    ]
    policy = BuyAndHold()
    out = tmp_path / "case_study.png"
    generate_event_case_study(bars, policy, "test_event", window_bars=20, start_bar=2, output_path=out)
    assert out.exists()
