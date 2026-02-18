"""Risk constraints from configs/risk.yaml. Breach → flatten + terminate (kill switch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_risk_config() -> dict[str, Any]:
    path = CONFIG_DIR / "risk.yaml"
    if not path.exists():
        return _default_risk()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("risk", _default_risk())


def _default_risk() -> dict[str, Any]:
    return {
        "max_premium_at_risk": 50000.0,
        "max_vega": 200.0,
        "max_delta": 500.0,
        "max_open_contracts": 100,
        "max_daily_loss": 5000.0,
    }


def check_risk_breach(
    premium_at_risk: float,
    net_vega: float,
    net_delta: float,
    open_contracts: int,
    daily_pnl: float,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Returns (breach, reason). If breach, caller should flatten and terminate.
    """
    cfg = config or load_risk_config()
    if premium_at_risk > cfg.get("max_premium_at_risk", float("inf")):
        return True, "max_premium_at_risk"
    if abs(net_vega) > cfg.get("max_vega", float("inf")):
        return True, "max_vega"
    if abs(net_delta) > cfg.get("max_delta", float("inf")):
        return True, "max_delta"
    if open_contracts > cfg.get("max_open_contracts", float("inf")):
        return True, "max_open_contracts"
    if daily_pnl < -cfg.get("max_daily_loss", float("inf")):
        return True, "max_daily_loss"
    return False, ""
