"""Tests for SB3 training: split_bars_by_time, train_agent (minimal run)."""

from __future__ import annotations

import pytest

from src.agents.train_sb3 import (
    create_train_val_test_envs,
    split_bars_by_time,
    train_agent,
    DiscreteToBoxWrapper,
)
from src.envs.options_env import OptionsEnv


def _make_fake_bars(n: int = 60) -> list[dict]:
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


def test_split_bars_by_time_ratios():
    bars = _make_fake_bars(100)
    train, val, test = split_bars_by_time(bars, train_pct=0.70, val_pct=0.15)
    assert len(train) + len(val) + len(test) == 100
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_create_train_val_test_envs():
    bars = _make_fake_bars(100)
    train_env, val_env, test_env = create_train_val_test_envs(bars, underlying="SPY", train_pct=0.7, val_pct=0.15)
    assert train_env._bars == bars[:70]
    assert val_env._bars == bars[70:85]
    assert test_env._bars == bars[85:]


def test_discrete_to_box_wrapper():
    env = OptionsEnv(feature_bars=_make_fake_bars(30))
    wrapped = DiscreteToBoxWrapper(env)
    assert wrapped.action_space.shape == (4,)
    obs, _ = wrapped.reset(seed=0)
    action = wrapped.action_space.sample()
    obs2, r, term, trunc, info = wrapped.step(action)
    assert obs2.shape == obs.shape
    assert isinstance(r, float)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("stable_baselines3") is None,
    reason="stable-baselines3 not installed",
)
def test_train_agent_ppo_minimal():
    bars = _make_fake_bars(80)
    env = OptionsEnv(feature_bars=bars)
    model = train_agent("ppo", env, total_timesteps=100, seed=0)
    assert model is not None
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (4,) or len(action) == 4
