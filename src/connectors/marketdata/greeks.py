"""Compute Black-Scholes Greeks via py_vollib when provider does not supply them."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default risk-free rate (annualized)
DEFAULT_R = 0.05


def _years_to_expiry(snapshot_date: date, expiry: date) -> float:
    """Time to expiry in years; minimum 1e-4 to avoid div/NaN."""
    delta = (expiry - snapshot_date).days
    if delta <= 0:
        return 1e-4
    return delta / 365.0


def compute_greeks(
    *,
    flag: str,  # 'c' | 'p'
    S: float,
    K: float,
    t: float,
    sigma: float,
    r: float = DEFAULT_R,
) -> dict[str, float]:
    """
    Return delta, gamma, theta, vega (and optionally rho) using Black-Scholes.
    sigma: annualized IV as decimal (e.g. 0.25).
    t: time to expiry in years.
    """
    try:
        from py_vollib.black_scholes.greeks import analytical
    except ImportError:
        logger.warning("py_vollib not installed; cannot compute Greeks")
        return {}
    if sigma <= 0 or t <= 0:
        return {}
    try:
        delta = analytical.delta(flag, S, K, t, r, sigma)
        gamma = analytical.gamma(flag, S, K, t, r, sigma)
        theta = analytical.theta(flag, S, K, t, r, sigma)
        vega = analytical.vega(flag, S, K, t, r, sigma)
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except Exception as e:
        logger.debug("Greeks computation failed: %s", e)
        return {}
