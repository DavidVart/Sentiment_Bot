"""Gymnasium-compatible options RL environment; reads from feature_bars, uses portfolio/execution/reward/constraints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

from src.envs.constraints import check_risk_breach, load_risk_config
from src.envs.execution_sim import ExecutionSimulator, Fill
from src.envs.portfolio_constructor import (
    build_target_positions,
    expiry_days_from_action,
    size_scalar_from_action,
)
from src.envs.reward import check_drawdown_terminate, compute_reward
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_feature_bars_from_db(
    underlying: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Load feature_bars from PostgreSQL for the given underlying. Returns list of row dicts."""
    from src.db import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            if start_date and end_date:
                cur.execute(
                    """
                    SELECT underlying, ts, atm_iv_7d, atm_iv_14d, atm_iv_30d, iv_term_slope, iv_skew,
                           realized_vol_5d, realized_vol_10d, realized_vol_20d, realized_vol_60d, vix_close, options_gap_flag,
                           sent_news_asset, sent_social_asset, sent_macro_topic, sent_dispersion, sent_momentum, sent_volume, no_news_flag,
                           pm_p, pm_logit_p, pm_delta_p_1h, pm_delta_p_1d, pm_momentum, pm_vol_of_p, pm_time_to_event, pm_surprise_z, pm_gap_flag,
                           equity_return_1d, equity_realized_vol_20d
                    FROM feature_bars
                    WHERE underlying = %s AND ts >= %s AND ts <= %s
                    ORDER BY ts
                    LIMIT %s
                    """,
                    (underlying, start_date, end_date, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT underlying, ts, atm_iv_7d, atm_iv_14d, atm_iv_30d, iv_term_slope, iv_skew,
                           realized_vol_5d, realized_vol_10d, realized_vol_20d, realized_vol_60d, vix_close, options_gap_flag,
                           sent_news_asset, sent_social_asset, sent_macro_topic, sent_dispersion, sent_momentum, sent_volume, no_news_flag,
                           pm_p, pm_logit_p, pm_delta_p_1h, pm_delta_p_1d, pm_momentum, pm_vol_of_p, pm_time_to_event, pm_surprise_z, pm_gap_flag,
                           equity_return_1d, equity_realized_vol_20d
                    FROM feature_bars
                    WHERE underlying = %s
                    ORDER BY ts
                    LIMIT %s
                    """,
                    (underlying, limit),
                )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]

# Observation dims: vol_regime 10 + options_surface 8 + portfolio 10 + sentiment 8 + pm 8*2 = 52
OBS_VOL = 10
OBS_OPTIONS = 8
OBS_PORTFOLIO = 10
OBS_SENTIMENT = 8
OBS_PM_PER_EVENT = 8
MAX_PM_EVENTS = 2
OBS_DIMS = OBS_VOL + OBS_OPTIONS + OBS_PORTFOLIO + OBS_SENTIMENT + OBS_PM_PER_EVENT * MAX_PM_EVENTS


def _normalize(x: float | None, low: float = -1.0, high: float = 1.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip(x, low, high))


def _zclip(x: float | None, scale: float = 1.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip(x / (scale + 1e-8), -1.0, 1.0))


def _load_execution_config() -> dict[str, Any]:
    path = CONFIG_DIR / "execution.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    ex = data.get("execution", {})
    rew = data.get("reward", {})
    ep = data.get("episode", {})
    return {
        **ex,
        "lambda_transaction_cost": rew.get("lambda_transaction_cost", 1.0),
        "lambda_drawdown_increment": rew.get("lambda_drawdown_increment", 2.0),
        "max_drawdown_pct": ep.get("max_drawdown_pct", 0.10),
    }


def _row_to_obs(
    row: dict[str, Any],
    cash: float,
    margin: float,
    net_delta: float,
    net_gamma: float,
    net_vega: float,
    net_theta: float,
    position_count: int,
    pm_events: list[dict[str, Any]],
) -> np.ndarray:
    """Build flat observation vector; normalize to [-1, 1] or z-score style."""
    obs = np.zeros(OBS_DIMS, dtype=np.float32)
    idx = 0
    # Vol regime ~10
    obs[idx] = _zclip(row.get("equity_return_1d"), 0.1)
    idx += 1
    for k in ("realized_vol_5d", "realized_vol_10d", "realized_vol_20d", "realized_vol_60d"):
        obs[idx] = _zclip(row.get(k), 0.5)
        idx += 1
    obs[idx] = _zclip(row.get("vix_close"), 50.0)
    idx += 1
    obs[idx] = _zclip(row.get("iv_term_slope"), 0.1)
    idx += 1
    obs[idx] = _zclip(row.get("iv_skew"), 0.2)
    idx += 1
    ret = row.get("equity_return_1d") or 0.0
    obs[idx] = 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)
    idx += 1
    obs[idx] = 1.0 if row.get("options_gap_flag") else -1.0
    idx += 1
    # Options surface ~8
    for k in ("atm_iv_7d", "atm_iv_14d", "atm_iv_30d"):
        obs[idx] = _zclip(row.get(k), 0.5)
        idx += 1
    obs[idx] = _zclip(row.get("iv_term_slope"), 0.1)
    idx += 1
    obs[idx] = _zclip(row.get("iv_skew"), 0.2)
    idx += 1
    obs[idx] = 0.0  # spread proxy placeholder
    idx += 1
    obs[idx] = 0.0
    idx += 1
    obs[idx] = 0.0
    idx += 1
    # Portfolio ~10
    obs[idx] = _zclip(cash, 100000.0)
    idx += 1
    obs[idx] = _zclip(margin, 50000.0)
    idx += 1
    obs[idx] = _zclip(net_delta, 500.0)
    idx += 1
    obs[idx] = _zclip(net_gamma, 50.0)
    idx += 1
    obs[idx] = _zclip(net_vega, 200.0)
    idx += 1
    obs[idx] = _zclip(net_theta, 500.0)
    idx += 1
    obs[idx] = _zclip(float(position_count), 100.0)
    idx += 1
    for _ in range(3):
        obs[idx] = 0.0
        idx += 1
    # Sentiment ~8
    obs[idx] = _normalize(row.get("sent_news_asset", 0), -1, 1)
    idx += 1
    obs[idx] = _normalize(row.get("sent_social_asset", 0), -1, 1)
    idx += 1
    obs[idx] = _normalize(row.get("sent_macro_topic", 0), -1, 1)
    idx += 1
    obs[idx] = _zclip(row.get("sent_dispersion", 0), 0.5)
    idx += 1
    obs[idx] = _normalize(row.get("sent_momentum", 0), -1, 1)
    idx += 1
    obs[idx] = _zclip(row.get("sent_volume", 0), 100.0)
    idx += 1
    obs[idx] = 1.0 if row.get("no_news_flag") else -1.0
    idx += 1
    obs[idx] = 0.0
    idx += 1
    # PM 8 per event (max 2)
    for i in range(MAX_PM_EVENTS):
        ev = pm_events[i] if i < len(pm_events) else {}
        obs[idx] = _normalize(ev.get("p"), 0, 1)
        idx += 1
        obs[idx] = _zclip(ev.get("logit_p"), 2.0)
        idx += 1
        obs[idx] = _zclip(ev.get("delta_p_1h"), 0.2)
        idx += 1
        obs[idx] = _zclip(ev.get("delta_p_1d"), 0.2)
        idx += 1
        obs[idx] = _zclip(ev.get("vol_of_p"), 0.2)
        idx += 1
        obs[idx] = _zclip(ev.get("surprise_z"), 2.0)
        idx += 1
        obs[idx] = _zclip(ev.get("time_to_event"), 720.0)  # hours
        idx += 1
        obs[idx] = 0.0  # cross_platform_spread placeholder
        idx += 1
    assert idx == OBS_DIMS
    return obs


class OptionsEnv(Env[np.ndarray, np.ndarray]):
    """Gymnasium env: feature_bars table, MultiDiscrete action, Box observation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_bars: list[dict[str, Any]],
        underlying: str = "SPY",
        initial_cash: float = 100_000.0,
        config: dict[str, Any] | None = None,
    ):
        """Create an options-trading env from a list of feature_bars (row dicts). Action space MultiDiscrete([3,3,3,3]); observation space Box(52,)."""
        super().__init__()
        self._bars = feature_bars
        self._underlying = underlying
        self._initial_cash = initial_cash
        self._config = config or _load_execution_config()
        self._risk_config = load_risk_config()
        self.observation_space = Box(low=-1.0, high=1.0, shape=(OBS_DIMS,), dtype=np.float32)
        self.action_space = MultiDiscrete([3, 3, 3, 3])  # vega, delta, size, expiry
        self._exec = ExecutionSimulator(self._config)
        self._bar_idx = 0
        self._cash = initial_cash
        self._positions: dict[str, int] = {}
        self._peak_equity = initial_cash
        self._daily_pnl_start = initial_cash
        self._day_start_bar = 0
        self._pm_events: list[dict[str, Any]] = []

    def _current_row(self) -> dict[str, Any]:
        if not self._bars or self._bar_idx >= len(self._bars):
            return {}
        return self._bars[self._bar_idx]

    def _equity(self) -> float:
        # Simplified: equity = cash + PnL from positions (we don't mark-to-market here; use cash + margin proxy)
        return self._cash

    def _portfolio_greeks(self) -> tuple[float, float, float, float]:
        # Placeholder: no real greeks from positions; use 0 or simple proxy
        net_delta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        net_theta = 0.0
        for _key, qty in self._positions.items():
            net_vega += qty * 10.0
            net_delta += qty * 5.0
        return net_delta, net_gamma, net_vega, net_theta

    def _apply_fills(self, fills: list[Fill]) -> float:
        pnl = 0.0
        for f in fills:
            self._positions[f.contract_key] = self._positions.get(f.contract_key, 0) + f.qty
            self._cash -= f.price * f.qty
            self._cash -= f.fee
            pnl -= f.fee
        return pnl

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the first bar; reinitialize cash, positions, and execution. Returns (observation, info)."""
        super().reset(seed=seed)
        self._bar_idx = 0
        self._cash = self._initial_cash
        self._positions = {}
        self._peak_equity = self._initial_cash
        self._daily_pnl_start = self._initial_cash
        self._day_start_bar = 0
        self._exec = ExecutionSimulator(self._config)
        self._pm_events = [{}] * MAX_PM_EVENTS
        obs = self._get_obs()
        return obs, {"bar_idx": 0}

    def _get_obs(self) -> np.ndarray:
        row = self._current_row()
        nd, ng, nv, nt = self._portfolio_greeks()
        pos_count = sum(1 for q in self._positions.values() if q != 0)
        return _row_to_obs(
            row,
            self._cash,
            0.0,
            nd,
            ng,
            nv,
            nt,
            pos_count,
            self._pm_events,
        )

    def step(
        self,
        action: np.ndarray | list[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step: interpret action (vega, delta, size, expiry), submit orders, advance bar, return (obs, reward, terminated, truncated, info) with info containing pnl, equity, net_delta, net_vega, vix, etc."""
        action = np.asarray(action).flatten()
        vega_idx = int(action[0]) if len(action) > 0 else 1
        delta_idx = int(action[1]) if len(action) > 1 else 1
        size_idx = int(action[2]) if len(action) > 2 else 2
        expiry_idx = int(action[3]) if len(action) > 3 else 1
        size_scalar = size_scalar_from_action(size_idx)
        expiry_days = expiry_days_from_action(expiry_idx)
        row = self._current_row()
        # Target positions and submit orders (mock mid/spread/volume)
        targets = build_target_positions(
            self._underlying,
            vega_idx,
            delta_idx,
            size_scalar,
            expiry_days,
            base_lots=1,
        )
        mid = 5.0
        spread = 0.10
        vol = 100.0
        for ck, qty in targets:
            self._exec.submit_order(ck, qty, mid, spread, vol, self._bar_idx)
        # Advance bar and process fills
        equity_before = self._equity()
        self._bar_idx += 1
        fills = self._exec.advance_bar(self._bar_idx)
        self._apply_fills(fills)
        equity_after = self._equity()
        pnl_step = equity_after - equity_before
        transaction_costs = sum(f.fee for f in fills)
        reward, self._peak_equity, _ = compute_reward(
            pnl_step,
            transaction_costs,
            self._peak_equity,
            equity_before,
            equity_after,
            self._config.get("lambda_transaction_cost", 1.0),
            self._config.get("lambda_drawdown_increment", 2.0),
        )
        nd, ng, nv, nt = self._portfolio_greeks()
        premium_at_risk = abs(nv) * 10.0
        daily_pnl = self._equity() - self._daily_pnl_start
        breach, reason = check_risk_breach(
            premium_at_risk,
            nv,
            nd,
            sum(abs(q) for q in self._positions.values()),
            daily_pnl,
            self._risk_config,
        )
        terminated = False
        if breach:
            terminated = True
            reward = -100.0
        if check_drawdown_terminate(
            self._peak_equity,
            self._equity(),
            self._config.get("max_drawdown_pct", 0.10),
        ):
            terminated = True
        truncated = self._bar_idx >= len(self._bars)
        obs = self._get_obs()
        row = self._current_row()
        info = {
            "bar_idx": self._bar_idx,
            "pnl": pnl_step,
            "equity": self._equity(),
            "risk_breach": reason if breach else None,
            "transaction_costs": transaction_costs,
            "net_delta": nd,
            "net_vega": nv,
            "vix": float(row.get("vix_close") or 0.0),
        }
        return obs, float(reward), terminated, truncated, info
