"""Tests for evaluation harness: compute_metrics, regime_split, bootstrap_sharpe."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.eval import (
    bootstrap_sharpe,
    compute_metrics,
    regime_split,
    evaluate_policy,
    evaluate_policy_with_series,
    walk_forward_evaluate,
)
from src.agents.baselines import BuyAndHold
from src.envs.options_env import OptionsEnv


def _make_fake_bars(n: int = 100) -> list[dict]:
    return [
        {
            "underlying": "SPY",
            "equity_return_1d": 0.001,
            "realized_vol_5d": 0.15,
            "realized_vol_10d": 0.16,
            "realized_vol_20d": 0.17,
            "realized_vol_60d": 0.18,
            "vix_close": 20.0,
            "iv_term_slope": -0.01,
            "iv_skew": 0.02,
            "options_gap_flag": False,
            "atm_iv_7d": 0.20,
            "atm_iv_14d": 0.19,
            "atm_iv_30d": 0.18,
            "sent_news_asset": 0.1,
            "sent_social_asset": 0.0,
            "sent_macro_topic": -0.05,
            "sent_dispersion": 0.02,
            "sent_momentum": 0.01,
            "sent_volume": 5,
            "no_news_flag": False,
        }
        for _ in range(n)
    ]


def test_compute_metrics_empty():
    m = compute_metrics(
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    )
    assert m["n_bars"] == 0
    assert m["annualized_sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0


def test_compute_metrics_known_equity_path():
    # Flat equity -> zero returns -> zero Sharpe
    n = 50
    equity = np.ones(n) * 100_000.0
    pnl = np.zeros(n)
    pnl[0] = 0.0
    costs = np.zeros(n)
    delta = np.zeros(n)
    vega = np.zeros(n)
    m = compute_metrics(pnl, equity, costs, delta, vega)
    assert m["annualized_sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["hit_rate_pct"] == 0.0
    assert m["n_bars"] == n


def test_compute_metrics_positive_returns():
    n = 100
    # Constant positive pnl each bar
    equity = 100_000.0 + np.cumsum(np.ones(n) * 10.0)
    pnl = np.ones(n) * 10.0
    costs = np.zeros(n)
    delta = np.ones(n) * 5.0
    vega = np.ones(n) * 10.0
    m = compute_metrics(pnl, equity, costs, delta, vega)
    assert m["annualized_sharpe"] > 0
    assert m["hit_rate_pct"] == 100.0
    # profit_factor is 0 when there are no losses (no denominator)
    assert m["total_pnl"] == 1000.0


def test_compute_metrics_drawdown():
    # Equity: 100 -> 110 -> 95 -> 105
    equity = np.array([100.0, 110.0, 95.0, 105.0])
    pnl = np.array([0.0, 10.0, -15.0, 10.0])
    costs = np.zeros(4)
    delta = np.zeros(4)
    vega = np.zeros(4)
    m = compute_metrics(pnl, equity, costs, delta, vega)
    assert m["max_drawdown"] > 0
    assert m["max_drawdown"] <= 1.0


def test_compute_metrics_avg_win_loss_ratio():
    pnl = np.array([10.0, -5.0, 20.0, -10.0, 15.0])
    equity = 100_000.0 + np.cumsum(pnl)
    equity = np.concatenate([[100_000.0], equity])[:-1]  # shift so equity[t-1] used
    equity = 100_000.0 + np.cumsum(pnl)  # equity after each step
    costs = np.zeros(5)
    delta = np.zeros(5)
    vega = np.zeros(5)
    m = compute_metrics(pnl, equity, costs, delta, vega)
    # Wins: 10, 20, 15 -> avg 15. Losses: 5, 10 -> avg 7.5. Ratio = 15/7.5 = 2.0
    assert m["avg_win_avg_loss_ratio"] == pytest.approx(2.0, rel=0.01)
    assert m["profit_factor"] == pytest.approx(45.0 / 15.0, rel=0.01)


def test_regime_split_empty_series():
    results = {"vix_series": [], "pnl_series": [], "equity_series": [], "transaction_costs_series": [], "net_delta_series": [], "net_vega_series": []}
    regimes = regime_split(results, vix_thresholds=[15, 25])
    assert "low" in regimes and "medium" in regimes and "high" in regimes


def test_regime_split_with_series():
    vix = np.array([10.0, 20.0, 30.0, 12.0, 26.0])
    pnl = np.ones(5) * 100.0
    equity = 100_000.0 + np.cumsum(pnl)
    results = {
        "vix_series": vix.tolist(),
        "pnl_series": pnl.tolist(),
        "equity_series": equity.tolist(),
        "transaction_costs_series": np.zeros(5).tolist(),
        "net_delta_series": np.zeros(5).tolist(),
        "net_vega_series": np.zeros(5).tolist(),
    }
    regimes = regime_split(results, vix_thresholds=[15, 25])
    assert regimes["low"]["n_bars"] == 2  # 10, 12
    assert regimes["medium"]["n_bars"] == 1  # 20
    assert regimes["high"]["n_bars"] == 2   # 30, 26


def test_bootstrap_sharpe_empty():
    point, lower, upper = bootstrap_sharpe(np.array([]), np.array([]), n_resamples=100)
    assert point == 0.0 and lower == 0.0 and upper == 0.0


def test_bootstrap_sharpe_returns_ci():
    np.random.seed(42)
    n = 200
    pnl = np.random.randn(n) * 10.0 + 1.0  # positive drift
    equity = 100_000.0 + np.cumsum(pnl)
    point, lower, upper = bootstrap_sharpe(pnl, equity, n_resamples=100, seed=43)
    assert lower <= point <= upper or (point == 0.0 and lower == 0.0 and upper == 0.0)


def test_evaluate_policy_runs():
    env = OptionsEnv(feature_bars=_make_fake_bars(80), underlying="SPY")
    policy = BuyAndHold()
    metrics = evaluate_policy(env, policy, n_episodes=1, seeds=[0])
    assert "annualized_sharpe" in metrics
    assert "total_pnl" in metrics
    assert "n_bars" in metrics
    assert metrics["n_bars"] > 0


def test_evaluate_policy_with_series_includes_series_and_ci():
    env = OptionsEnv(feature_bars=_make_fake_bars(80), underlying="SPY")
    policy = BuyAndHold()
    metrics = evaluate_policy_with_series(env, policy, n_episodes=1, seeds=[0])
    assert "pnl_series" in metrics
    assert "equity_series" in metrics
    assert "vix_series" in metrics
    assert "sharpe_ci95_lower" in metrics
    assert "sharpe_ci95_upper" in metrics
    assert len(metrics["pnl_series"]) == metrics["n_bars"]


def test_walk_forward_evaluate():
    def env_factory():
        return OptionsEnv(feature_bars=_make_fake_bars(60), underlying="SPY")

    policy = BuyAndHold()
    results = walk_forward_evaluate(env_factory, policy, n_windows=2, seeds=[0, 1])
    assert len(results) == 2
    assert "annualized_sharpe" in results[0]
    assert "n_bars" in results[0]
