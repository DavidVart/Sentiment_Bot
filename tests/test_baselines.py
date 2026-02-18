"""Tests for baseline policies: select_action shape, valid range, and rule logic."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.baselines import (
    BuyAndHold,
    DeltaNeutral,
    FixedLongVol,
    OBS_PM_DELTA_1H_0,
    OBS_PM_DELTA_1H_1,
    RandomPolicy,
    SimpleEventRule,
)


def _obs_52():
    return np.zeros(52, dtype=np.float32)


def test_buy_and_hold_action_shape_and_values():
    policy = BuyAndHold()
    obs = _obs_52()
    action = policy.select_action(obs)
    assert action.shape == (4,)
    assert action.dtype in (np.int64, np.int32, int)
    assert np.all((action >= 0) & (action <= 2))
    assert action[0] == 1 and action[1] == 1  # vega=0, delta=0


def test_fixed_long_vol_action():
    policy = FixedLongVol()
    action = policy.select_action(_obs_52())
    assert action.shape == (4,)
    assert action[0] == 2 and action[1] == 1  # vega=+1, delta=0
    assert action[2] == 2 and action[3] == 2  # size=1.0, expiry=30D


def test_delta_neutral_action():
    policy = DeltaNeutral()
    action = policy.select_action(_obs_52())
    assert action.shape == (4,)
    assert action[0] == 2 and action[1] == 1
    assert np.all((action >= 0) & (action <= 2))


def test_simple_event_rule_flat_when_below_threshold():
    policy = SimpleEventRule(delta_p_1h_threshold=0.05)
    obs = _obs_52()
    obs[OBS_PM_DELTA_1H_0] = 0.02
    obs[OBS_PM_DELTA_1H_1] = 0.01
    action = policy.select_action(obs)
    assert action[0] == 1 and action[1] == 1  # flat


def test_simple_event_rule_straddle_when_above_threshold():
    policy = SimpleEventRule(delta_p_1h_threshold=0.05)
    obs = _obs_52()
    obs[OBS_PM_DELTA_1H_0] = 0.10
    obs[OBS_PM_DELTA_1H_1] = 0.01
    action = policy.select_action(obs)
    assert action[0] == 2 and action[1] == 1  # straddle

    obs[OBS_PM_DELTA_1H_0] = 0.01
    obs[OBS_PM_DELTA_1H_1] = 0.08
    action = policy.select_action(obs)
    assert action[0] == 2 and action[1] == 1


def test_simple_event_rule_custom_threshold():
    policy = SimpleEventRule(delta_p_1h_threshold=0.10)
    obs = _obs_52()
    obs[OBS_PM_DELTA_1H_0] = 0.06
    action = policy.select_action(obs)
    assert action[0] == 1 and action[1] == 1  # below 0.10
    obs[OBS_PM_DELTA_1H_0] = 0.12
    action = policy.select_action(obs)
    assert action[0] == 2 and action[1] == 1


def test_simple_event_rule_short_obs():
    policy = SimpleEventRule()
    obs = np.zeros(10, dtype=np.float32)  # too short for PM indices
    action = policy.select_action(obs)
    assert action.shape == (4,)
    assert action[0] == 1 and action[1] == 1  # flat


def test_random_policy_action_shape_and_range():
    policy = RandomPolicy(seed=42)
    obs = _obs_52()
    actions = [policy.select_action(obs) for _ in range(50)]
    for a in actions:
        assert a.shape == (4,)
        assert np.all((a >= 0) & (a <= 2))
    # Should get some variation (not all same)
    arr = np.array(actions)
    assert np.any(arr[:, 0] != arr[0, 0]) or np.any(arr[:, 1] != arr[0, 1])


def test_random_policy_deterministic_with_same_seed():
    p1 = RandomPolicy(seed=99)
    p2 = RandomPolicy(seed=99)
    obs = _obs_52()
    a1 = [p1.select_action(obs) for _ in range(5)]
    a2 = [p2.select_action(obs) for _ in range(5)]
    for x, y in zip(a1, a2):
        np.testing.assert_array_equal(x, y)
