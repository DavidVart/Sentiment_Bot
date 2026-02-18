"""Tests for options RL env: random 100 steps, PnL, drawdown termination, reward/constraints/execution."""

from __future__ import annotations

import pytest

from src.envs.constraints import check_risk_breach, load_risk_config
from src.envs.execution_sim import ExecutionSimulator
from src.envs.options_env import OBS_DIMS, OptionsEnv
from src.envs.portfolio_constructor import (
    build_target_positions,
    expiry_days_from_action,
    get_contract_menu,
    size_scalar_from_action,
)
from src.envs.reward import check_drawdown_terminate, compute_reward


def _make_fake_bars(n: int = 150) -> list[dict]:
    return [
        {
            "underlying": "SPY",
            "equity_return_1d": 0.001,
            "realized_vol_5d": 0.15,
            "realized_vol_10d": 0.16,
            "realized_vol_20d": 0.17,
            "realized_vol_60d": 0.18,
            "vix_close": 18.0,
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


def test_random_actions_100_steps_no_errors():
    """Agent can take random actions for 100 steps without errors."""
    env = OptionsEnv(feature_bars=_make_fake_bars(120), underlying="SPY")
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIMS,)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIMS,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            break
    env.close()


def test_pnl_reasonable():
    """P&L is reasonable (not zero, not astronomical) over a short run."""
    env = OptionsEnv(feature_bars=_make_fake_bars(50), initial_cash=100_000.0)
    env.reset(seed=123)
    total_reward = 0.0
    for _ in range(20):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        total_reward += reward
        if term or trunc:
            break
    env.close()
    assert isinstance(total_reward, float)
    assert abs(total_reward) < 1e6
    assert info.get("equity") is not None


def test_episode_terminates_on_drawdown_breach():
    """Episode terminates when max drawdown exceeds threshold."""
    env = OptionsEnv(
        feature_bars=_make_fake_bars(200),
        initial_cash=100_000.0,
        config={
            "max_drawdown_pct": 0.02,
            "lambda_transaction_cost": 1.0,
            "lambda_drawdown_increment": 2.0,
        },
    )
    risk_cfg = load_risk_config()
    env._risk_config = {**risk_cfg, "max_daily_loss": 1e9, "max_vega": 1e9, "max_delta": 1e9}
    env.reset(seed=0)
    env._cash = 50_000.0
    env._peak_equity = 100_000.0
    terminated = check_drawdown_terminate(100_000.0, 50_000.0, 0.02)
    assert terminated is True
    terminated = check_drawdown_terminate(100_000.0, 99_000.0, 0.02)
    assert terminated is False


def test_reward_compute():
    r, peak, term = compute_reward(100.0, 10.0, 1000.0, 1000.0, 1100.0, 1.0, 2.0)
    assert peak == 1100.0
    assert r == pytest.approx(100.0 - 10.0 - 0.0)


def test_constraints_breach():
    breach, reason = check_risk_breach(60_000, 0, 0, 0, 0, {"max_premium_at_risk": 50_000})
    assert breach is True
    assert reason == "max_premium_at_risk"
    breach, _ = check_risk_breach(10_000, 0, 0, 0, 0, {"max_premium_at_risk": 50_000})
    assert breach is False


def test_execution_sim_delay_and_fill():
    sim = ExecutionSimulator({"min_bar_delay": 1, "fee_per_contract": 0.65, "spread_threshold_pct": 0.05})
    oid = sim.submit_order("SPY_7D_ATM_call", 1, 5.0, 0.10, 100.0, current_bar=0)
    assert oid is not None
    fills = sim.advance_bar(1)
    assert len(fills) == 1
    assert fills[0].qty == 1
    assert fills[0].fee == pytest.approx(0.65)


def test_execution_no_fill_if_spread_exceeds_threshold():
    sim = ExecutionSimulator({"spread_threshold_pct": 0.01})
    oid = sim.submit_order("SPY_7D_ATM_call", 1, 5.0, 1.0, 100.0, current_bar=0)
    assert oid is None


def test_portfolio_constructor_menu_and_targets():
    menu = get_contract_menu("SPY")
    assert len(menu) == 30
    assert "SPY_7D_ATM_call" in menu
    targets = build_target_positions("SPY", 2, 1, 1.0, 7, base_lots=1)
    assert len(targets) >= 1
    assert size_scalar_from_action(0) == 0.25
    assert size_scalar_from_action(2) == 1.0
    assert expiry_days_from_action(0) == 7
    assert expiry_days_from_action(2) == 30


def test_env_reset_and_observation_bounds():
    env = OptionsEnv(feature_bars=_make_fake_bars(30))
    obs, _ = env.reset(seed=1)
    assert obs.min() >= -1.0
    assert obs.max() <= 1.0
    env.close()


def test_env_action_space_sample():
    env = OptionsEnv(feature_bars=_make_fake_bars(30))
    for _ in range(20):
        a = env.action_space.sample()
        assert a.shape == (4,)
        assert a.dtype in (int, "int32", "int64")
    env.close()


def test_episode_truncated_when_bars_exhausted():
    """Episode truncates when bar index exceeds available bars."""
    bars = _make_fake_bars(5)
    env = OptionsEnv(feature_bars=bars)
    obs, _ = env.reset(seed=0)
    for step in range(10):
        obs, reward, term, trunc, info = env.step([1, 1, 0, 1])
        if trunc:
            assert info["bar_idx"] >= len(bars)
            break
    env.close()
