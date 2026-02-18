"""Tests for observation mask wrapper and variant masks."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.obs_mask_wrapper import (
    OBS_DIMS,
    OBS_OPTIONS_END,
    OBS_PM_END,
    OBS_PORTFOLIO_END,
    OBS_SENTIMENT_END,
    OBS_VOL_END,
    ObsMaskWrapper,
    VARIANT_MASKS,
    make_mask,
)
from src.envs.options_env import OptionsEnv


def _make_fake_bars(n: int = 30) -> list[dict]:
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


def test_make_mask_full():
    m = make_mask(include_vol=True, include_options=True, include_portfolio=True, include_sentiment=True, include_pm=True)
    assert m.shape == (OBS_DIMS,)
    assert np.all(m == 1.0)


def test_make_mask_base():
    m = make_mask(include_vol=True, include_options=True, include_portfolio=True, include_sentiment=False, include_pm=False)
    assert m.shape == (OBS_DIMS,)
    assert np.all(m[0:OBS_PORTFOLIO_END] == 1.0)
    assert np.all(m[OBS_PORTFOLIO_END:] == 0.0)


def test_variant_masks_exist():
    for k in ("A", "B", "C", "D"):
        assert k in VARIANT_MASKS
        assert VARIANT_MASKS[k].shape == (OBS_DIMS,)
        assert np.all((VARIANT_MASKS[k] == 0) | (VARIANT_MASKS[k] == 1))


def test_variant_a_zeroes_sentiment_and_pm():
    m = VARIANT_MASKS["A"]
    assert np.all(m[0:OBS_PORTFOLIO_END] == 1.0)
    assert np.all(m[OBS_PORTFOLIO_END:OBS_SENTIMENT_END] == 0.0)
    assert np.all(m[OBS_SENTIMENT_END:] == 0.0)


def test_variant_d_all_ones():
    assert np.all(VARIANT_MASKS["D"] == 1.0)


def test_obs_mask_wrapper_shape_unchanged():
    env = OptionsEnv(feature_bars=_make_fake_bars(20))
    wrapped = ObsMaskWrapper(env, variant="A")
    assert wrapped.observation_space.shape == env.observation_space.shape
    obs, _ = wrapped.reset(seed=0)
    assert obs.shape == (OBS_DIMS,)


def test_obs_mask_wrapper_zeroes_masked():
    env = OptionsEnv(feature_bars=_make_fake_bars(20))
    wrapped = ObsMaskWrapper(env, variant="A")
    obs, _ = wrapped.reset(seed=0)
    assert np.all(obs[OBS_PORTFOLIO_END:] == 0.0)
    obs2, _, _, _, _ = wrapped.step(env.action_space.sample())
    assert np.all(obs2[OBS_PORTFOLIO_END:] == 0.0)


def test_obs_mask_wrapper_step_returns_masked_obs():
    env = OptionsEnv(feature_bars=_make_fake_bars(25))
    wrapped = ObsMaskWrapper(env, variant="B")
    wrapped.reset(seed=1)
    obs, _, _, _, _ = wrapped.step([1, 1, 0, 0])
    assert obs.shape == (OBS_DIMS,)
    assert np.all(obs[OBS_SENTIMENT_END:] == 0.0)
