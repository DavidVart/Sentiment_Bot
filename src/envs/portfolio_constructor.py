"""Translates exposure targets (vega/delta buckets, size, expiry) into specific contracts from restricted menu."""

from __future__ import annotations

from typing import Any

EXPIRY_DAYS = (7, 14, 30)
STRIKE_TYPES = ("ATM", "M1", "P1", "M2", "P2")  # ATM, minus 1 sigma, plus 1 sigma, etc.
SIZE_SCALARS = (0.25, 0.5, 1.0)


def contract_key(underlying: str, expiry_days: int, strike_type: str, option_type: str) -> str:
    return f"{underlying}_{expiry_days}D_{strike_type}_{option_type}"


def get_contract_menu(underlying: str) -> list[str]:
    """Restricted menu: ~7D/14D/30D × ATM/±1σ/±2σ × call/put (~30 contracts)."""
    keys: list[str] = []
    for exp in EXPIRY_DAYS:
        for strike in STRIKE_TYPES:
            for opt in ("call", "put"):
                keys.append(contract_key(underlying, exp, strike, opt))
    return keys


def _strategy_legs(vega_bucket: int, delta_bucket: int) -> list[tuple[str, str, int]]:
    """
    Maps (vega_bucket, delta_bucket) to list of (strike_type, option_type, sign).
    vega_bucket: 0=-1, 1=0, 2=+1; delta_bucket: 0=-1, 1=0, 2=+1.
    +1 vega, 0 delta → long straddle (ATM call + ATM put)
    -1 vega, 0 delta → iron condor (sell OTM put spread + OTM call spread) → short P2 put, long P1 put, long M1 call, short M2 call
    """
    legs: list[tuple[str, str, int]] = []
    if vega_bucket == 2 and delta_bucket == 1:  # +vega, 0 delta
        legs = [("ATM", "call", 1), ("ATM", "put", 1)]
    elif vega_bucket == 0 and delta_bucket == 1:  # -vega, 0 delta
        legs = [("P2", "put", -1), ("P1", "put", 1), ("M1", "call", 1), ("M2", "call", -1)]
    elif vega_bucket == 1 and delta_bucket == 2:  # 0 vega, +delta
        legs = [("ATM", "call", 1), ("P1", "call", -1)]
    elif vega_bucket == 1 and delta_bucket == 0:  # 0 vega, -delta
        legs = [("ATM", "put", 1), ("M1", "put", -1)]
    elif vega_bucket == 2 and delta_bucket == 2:  # +vega, +delta
        legs = [("ATM", "call", 1), ("ATM", "put", 1), ("P1", "put", -1)]
    elif vega_bucket == 2 and delta_bucket == 0:  # +vega, -delta
        legs = [("ATM", "call", 1), ("ATM", "put", 1), ("M1", "call", -1)]
    elif vega_bucket == 0 and delta_bucket == 2:  # -vega, +delta
        legs = [("M1", "call", 1), ("M2", "call", -1)]
    elif vega_bucket == 0 and delta_bucket == 0:  # -vega, -delta
        legs = [("P1", "put", -1), ("P2", "put", 1)]
    else:  # 0 vega, 0 delta
        legs = []
    return legs


def size_scalar_from_action(action_size_idx: int) -> float:
    """action_size_idx 0, 1, 2 → 0.25, 0.5, 1.0."""
    return SIZE_SCALARS[min(2, max(0, action_size_idx))]


def expiry_days_from_action(action_expiry_idx: int) -> int:
    """action_expiry_idx 0, 1, 2 → 7, 14, 30."""
    return EXPIRY_DAYS[min(2, max(0, action_expiry_idx))]


def build_target_positions(
    underlying: str,
    vega_bucket: int,
    delta_bucket: int,
    size_scalar: float,
    expiry_days: int,
    base_lots: int = 1,
) -> list[tuple[str, int]]:
    """
    Returns list of (contract_key, qty). qty is in contracts (positive = long, negative = short).
    base_lots scaled by size_scalar; rounded to int.
    """
    legs = _strategy_legs(vega_bucket, delta_bucket)
    out: list[tuple[str, int]] = []
    for strike_type, option_type, sign in legs:
        key = contract_key(underlying, expiry_days, strike_type, option_type)
        qty = round(base_lots * size_scalar * sign)
        if qty != 0:
            out.append((key, qty))
    return out
