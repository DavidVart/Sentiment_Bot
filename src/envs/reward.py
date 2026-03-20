"""
Reward function for options RL environment.

All dollar amounts are normalized as percentage of equity, so reward
components are comparable in magnitude regardless of portfolio size.

r_t = pnl_pct                                   (% return this bar)
      - lambda_cost * fee_pct                    (% cost from fees)
      - lambda_dd * drawdown_increment_pct       (% drawdown increase)
      ± inaction_penalty / exposure_bonus         (fixed-scale incentives)

Episode termination on drawdown breach.
"""

from __future__ import annotations


def compute_reward(
    pnl_step: float,
    transaction_costs: float,
    peak_before: float,
    equity_before: float,
    equity_after: float,
    lambda_cost: float = 1.0,
    lambda_dd: float = 1.0,
    has_positions: bool = False,
    inaction_penalty: float = 0.02,
    exposure_bonus: float = 0.01,
) -> tuple[float, float, bool]:
    """
    Returns (reward, new_peak, terminated_by_drawdown).

    All dollar values normalized to percentage of equity_before, so:
    - A $23 PnL on $100k = 0.023 reward (tiny, as expected for 1 bar)
    - A $3.25 fee on $100k = 0.00325 penalty
    - Inaction penalty 0.02/bar → over 600 bars = -12.0 total
    - Exposure bonus 0.01/bar → over 600 bars = +6.0 total
    - Net incentive to trade: 18.0 >> PnL noise std (~0.56)
    """
    normalizer = max(equity_before, 1.0)

    # Normalize to percentage of equity (1% move = reward of 1.0)
    pnl_pct = (pnl_step / normalizer) * 100.0
    fee_pct = (transaction_costs / normalizer) * 100.0

    new_peak = max(peak_before, equity_after)
    dd_before = max(0.0, peak_before - equity_before)
    dd_after = max(0.0, new_peak - equity_after)
    dd_increment = max(0.0, dd_after - dd_before)
    dd_pct = (dd_increment / normalizer) * 100.0

    reward = pnl_pct - lambda_cost * fee_pct - lambda_dd * dd_pct

    # Fixed-scale incentives to break the do-nothing equilibrium
    if has_positions:
        reward += exposure_bonus
    else:
        reward -= inaction_penalty

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
