"""Five baseline strategies for OptionsEnv; all implement select_action(observation) -> action."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

# Observation layout (from options_env): PM delta_p_1h at indices 38 and 46 (first and second event)
OBS_PM_DELTA_1H_0 = 38
OBS_PM_DELTA_1H_1 = 46


class Policy(Protocol):
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Return action array shape (4,) for MultiDiscrete [3,3,3,3]: vega, delta, size, expiry."""
        ...


class BuyAndHold:
    """Synthetic long SPY via bull call spread (0 vega, +delta). Holds steady position each bar."""

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return np.array([1, 2, 0, 1], dtype=np.int64)  # vega=0, delta=+1, size=0.25, expiry=14D


class FixedLongVol:
    """Every bar: buy ATM straddle 30D (vega=+1, delta=0, size=0.5, expiry=30D). Sized to stay within vega limits."""

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return np.array([2, 1, 1, 2], dtype=np.int64)  # vega=+1, delta=0, size=0.5, expiry=30D


class SimpleEventRule:
    """If any PM delta_p_1h > threshold (in obs space), buy straddle; else flat.
    Default threshold 0.01 (raw delta ~0.002 after z-clip with scale 0.2).
    Uses abs() to catch both positive and negative moves."""

    def __init__(self, delta_p_1h_threshold: float = 0.01):
        self.threshold = delta_p_1h_threshold

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation).flatten()
        if len(obs) <= OBS_PM_DELTA_1H_1:
            return np.array([1, 1, 0, 0], dtype=np.int64)
        if abs(obs[OBS_PM_DELTA_1H_0]) > self.threshold or abs(obs[OBS_PM_DELTA_1H_1]) > self.threshold:
            return np.array([2, 1, 1, 1], dtype=np.int64)  # straddle, 0.5 size, 14D
        return np.array([1, 1, 0, 0], dtype=np.int64)


class DeltaNeutral:
    """Hedge to zero delta, hold vega: vega=+1, delta=0 every bar."""

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return np.array([2, 1, 2, 1], dtype=np.int64)  # vega=+1, delta=0, size=1.0, expiry=14D


class RandomPolicy:
    """Uniformly random action from action space each step."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        return self._rng.integers(0, 3, size=4).astype(np.int64)
