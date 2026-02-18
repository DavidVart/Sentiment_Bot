"""Reward: r_t = pnl_t - λ1 * transaction_costs - λ2 * max_drawdown_increment. Episode end on drawdown breach."""

from __future__ import annotations

from typing import Any


def compute_reward(
    pnl_step: float,
    transaction_costs: float,
    peak_before: float,
    equity_before: float,
    equity_after: float,
    lambda_cost: float = 1.0,
    lambda_dd: float = 2.0,
) -> tuple[float, float, bool]:
    """
    Returns (reward, new_peak, terminated_by_drawdown).
    max_drawdown_increment = max(0, (peak - equity_after) - (peak - equity_before)) = max(0, equity_before - equity_after) when equity_after < peak.
    Simplification: dd_increment = max(0, peak_before - equity_after) - max(0, peak_before - equity_before).
    """
    new_peak = max(peak_before, equity_after)
    dd_before = max(0.0, peak_before - equity_before)
    dd_after = max(0.0, new_peak - equity_after)
    dd_increment = max(0.0, dd_after - dd_before)
    reward = pnl_step - lambda_cost * transaction_costs - lambda_dd * dd_increment
    return reward, new_peak, False


def check_drawdown_terminate(
    peak: float,
    equity: float,
    max_drawdown_pct: float,
) -> bool:
    """Terminate episode if current drawdown from peak exceeds threshold."""
    if peak <= 0:
        return False
    drawdown_pct = (peak - equity) / peak
    return drawdown_pct >= max_drawdown_pct
