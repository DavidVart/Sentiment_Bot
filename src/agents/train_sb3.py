"""Train PPO and SAC agents via Stable-Baselines3 with time-based train/val/test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from src.envs.options_env import OptionsEnv

# Lazy import so SB3 is optional at import time
def _get_sb3():
    import stable_baselines3
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    return stable_baselines3, PPO, SAC, DummyVecEnv


# Observation indices (from options_env): vol 0-9, options 10-17, portfolio 18-27, sentiment 28-35, PM 36-51
OBS_VOL_END = 10
OBS_OPTIONS_END = 18
OBS_PORTFOLIO_END = 28
OBS_SENTIMENT_END = 36
OBS_PM_END = 52


def split_bars_by_time(
    feature_bars: list[dict[str, Any]],
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split feature_bars by time only (no shuffling). Returns (train, val, test). test_pct = 1 - train_pct - val_pct."""
    n = len(feature_bars)
    if n == 0:
        return [], [], []
    t1 = int(n * train_pct)
    t2 = int(n * (train_pct + val_pct))
    train = feature_bars[:t1]
    val = feature_bars[t1:t2]
    test = feature_bars[t2:]
    return train, val, test


class DiscreteToBoxWrapper(Env):
    """Wraps an env with MultiDiscrete([3,3,3,3]) so action space is Box(0, 2, (4,)) for SAC."""

    def __init__(self, env: Env):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = Box(low=0.0, high=2.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        a = np.asarray(action).flatten()
        a_int = np.round(np.clip(a, 0.0, 2.0)).astype(np.int64)
        return self._env.step(a_int)

    def close(self):
        return self._env.close()


def train_agent(
    algorithm: str,
    env: Env,
    total_timesteps: int = 100_000,
    seed: int = 0,
    log_dir: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """
    Train a PPO or SAC agent. Uses MlpPolicy with 2 hidden layers of 256 units.
    SAC is wrapped via DiscreteToBoxWrapper (continuous action then rounded to discrete).
    """
    _, PPO_cls, SAC_cls, DummyVecEnv = _get_sb3()
    policy_kwargs = {"net_arch": dict(pi=[256, 256], vf=[256, 256])}  # PPO
    if algorithm.lower() == "sac":
        env = DiscreteToBoxWrapper(env)
        policy_kwargs = {"net_arch": [256, 256]}
    env = DummyVecEnv([lambda: env])

    log_dir = str(log_dir) if log_dir else None
    if algorithm.lower() == "ppo":
        model = PPO_cls(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
            tensorboard_log=log_dir,
            **kwargs,
        )
    elif algorithm.lower() == "sac":
        model = SAC_cls(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
            tensorboard_log=log_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"algorithm must be 'ppo' or 'sac', got {algorithm!r}")

    model.learn(total_timesteps=total_timesteps)
    return model


def create_train_val_test_envs(
    feature_bars: list[dict[str, Any]],
    underlying: str = "SPY",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[OptionsEnv, OptionsEnv, OptionsEnv]:
    """Create three OptionsEnvs for train, val, test by time split."""
    train_bars, val_bars, test_bars = split_bars_by_time(feature_bars, train_pct, val_pct)
    train_env = OptionsEnv(feature_bars=train_bars, underlying=underlying)
    val_env = OptionsEnv(feature_bars=val_bars, underlying=underlying)
    test_env = OptionsEnv(feature_bars=test_bars, underlying=underlying)
    return train_env, val_env, test_env
