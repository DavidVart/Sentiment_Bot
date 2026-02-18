"""Execution simulator: fill at mid ± half spread, slippage, fee, 1-bar delay, no fill if spread > threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class PendingOrder:
    order_id: int
    contract_key: str
    qty: int
    mid: float
    spread: float
    volume: float
    fill_at_bar: int


def _load_execution_config() -> dict[str, Any]:
    path = CONFIG_DIR / "execution.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    ex = data.get("execution", {})
    return {
        "fee_per_contract": float(ex.get("fee_per_contract", 0.65)),
        "min_bar_delay": int(ex.get("min_bar_delay", 1)),
        "spread_threshold_pct": float(ex.get("spread_threshold_pct", 0.05)),
        "slippage_bps_per_pct_volume": float(ex.get("slippage_bps_per_pct_volume", 2.0)),
    }


@dataclass
class Fill:
    order_id: int
    contract_key: str
    qty: int
    price: float
    fee: float


class ExecutionSimulator:
    """Min 1-bar delay; fill at mid ± half spread; slippage by size/volume; no fill if spread > threshold."""

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or _load_execution_config()
        self._pending: list[PendingOrder] = []
        self._next_order_id = 0
        self._current_bar = 0

    @property
    def fee_per_contract(self) -> float:
        return self._config.get("fee_per_contract", 0.65)

    @property
    def min_bar_delay(self) -> int:
        return self._config.get("min_bar_delay", 1)

    @property
    def spread_threshold_pct(self) -> float:
        return self._config.get("spread_threshold_pct", 0.05)

    def submit_order(
        self,
        contract_key: str,
        qty: int,
        mid: float,
        spread: float,
        volume: float,
        current_bar: int,
    ) -> int | None:
        """Submit order; returns order_id or None if spread exceeds threshold."""
        if qty == 0:
            return None
        spread_pct = spread / mid if mid and mid > 0 else 0.0
        if spread_pct > self.spread_threshold_pct:
            return None
        self._next_order_id += 1
        fill_at = current_bar + self.min_bar_delay
        self._pending.append(
            PendingOrder(
                order_id=self._next_order_id,
                contract_key=contract_key,
                qty=qty,
                mid=mid,
                spread=spread,
                volume=volume,
                fill_at_bar=fill_at,
            )
        )
        return self._next_order_id

    def advance_bar(self, current_bar: int) -> list[Fill]:
        """Process fills for current_bar (orders that had fill_at_bar == current_bar). Returns list of fills."""
        fills: list[Fill] = []
        still_pending: list[PendingOrder] = []
        for po in self._pending:
            if po.fill_at_bar != current_bar:
                still_pending.append(po)
                continue
            # Fill at mid ± half spread (sell = mid - half_spread, buy = mid + half_spread for negative qty convention or vice versa)
            half = po.spread / 2.0
            # qty > 0 = buy, qty < 0 = sell. Buy pays more (mid + half), sell receives less (mid - half)
            if po.qty > 0:
                fill_price = po.mid + half
            else:
                fill_price = po.mid - half
            # Slippage: proportional to spread and |qty|/volume
            slippage_pct = 0.0
            if po.volume and po.volume > 0:
                size_pct = min(1.0, abs(po.qty) / po.volume)
                bps = self._config.get("slippage_bps_per_pct_volume", 2.0) * size_pct * 100
                slippage_pct = bps / 10000.0
            if po.qty > 0:
                fill_price *= 1 + slippage_pct
            else:
                fill_price *= 1 - slippage_pct
            fee = self.fee_per_contract * abs(po.qty)
            fills.append(
                Fill(order_id=po.order_id, contract_key=po.contract_key, qty=po.qty, price=fill_price, fee=fee)
            )
        self._pending = still_pending
        return fills
