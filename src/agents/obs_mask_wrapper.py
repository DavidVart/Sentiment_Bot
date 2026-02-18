"""Observation mask wrapper: zero out feature groups to keep observation shape constant."""

from __future__ import annotations

import numpy as np
from gymnasium import Env

# Indices from options_env: vol 0-9, options 10-17, portfolio 18-27, sentiment 28-35, PM 36-51
OBS_VOL_END = 10
OBS_OPTIONS_END = 18
OBS_PORTFOLIO_END = 28
OBS_SENTIMENT_END = 36
OBS_PM_END = 52
OBS_DIMS = 52


def make_mask(
    include_vol: bool = True,
    include_options: bool = True,
    include_portfolio: bool = True,
    include_sentiment: bool = True,
    include_pm: bool = True,
) -> np.ndarray:
    """Build a 1/0 mask of shape (OBS_DIMS,) for which observation indices to keep."""
    mask = np.zeros(OBS_DIMS, dtype=np.float32)
    if include_vol:
        mask[0:OBS_VOL_END] = 1.0
    if include_options:
        mask[OBS_VOL_END:OBS_OPTIONS_END] = 1.0
    if include_portfolio:
        mask[OBS_OPTIONS_END:OBS_PORTFOLIO_END] = 1.0
    if include_sentiment:
        mask[OBS_PORTFOLIO_END:OBS_SENTIMENT_END] = 1.0
    if include_pm:
        mask[OBS_SENTIMENT_END:OBS_PM_END] = 1.0
    return mask


# Variant masks: Base = options + underlying (vol); +Sentiment = Base + sentiment; +PM = Base + PM; Full = all
VARIANT_MASKS = {
    "A": make_mask(include_vol=True, include_options=True, include_portfolio=True, include_sentiment=False, include_pm=False),
    "B": make_mask(include_vol=True, include_options=True, include_portfolio=True, include_sentiment=True, include_pm=False),
    "C": make_mask(include_vol=True, include_options=True, include_portfolio=True, include_sentiment=False, include_pm=True),
    "D": make_mask(include_vol=True, include_options=True, include_portfolio=True, include_sentiment=True, include_pm=True),
}


class ObsMaskWrapper(Env):
    """Zero out excluded observation dimensions; observation shape unchanged."""

    def __init__(self, env: Env, mask: np.ndarray | None = None, variant: str | None = None):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        if variant is not None:
            self._mask = VARIANT_MASKS.get(variant.upper(), np.ones(OBS_DIMS, dtype=np.float32))
        elif mask is not None:
            self._mask = np.asarray(mask, dtype=np.float32).flatten()
            if self._mask.shape[0] != OBS_DIMS:
                self._mask = np.ones(OBS_DIMS, dtype=np.float32)
        else:
            self._mask = np.ones(OBS_DIMS, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return (obs * self._mask).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (obs * self._mask).astype(np.float32), reward, terminated, truncated, info

    def close(self):
        return self._env.close()
