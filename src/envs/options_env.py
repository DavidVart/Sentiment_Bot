"""Gymnasium-compatible options RL environment; reads from feature_bars, uses portfolio/execution/reward/constraints."""

from __future__ import annotations

import math
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

# --------------------------------------------------------------------------- #
#  Spot reference price per underlying (used for BS pricing)                   #
# --------------------------------------------------------------------------- #
_SPOT_REF: dict[str, float] = {"SPY": 500.0, "QQQ": 430.0, "AAPL": 220.0}

# Strike offsets per strike_type (in units of sigma = spot * iv * sqrt(T))
_STRIKE_OFFSET: dict[str, float] = {
    "ATM": 0.0,
    "M1": -1.0,   # 1σ below (OTM put / ITM call)
    "P1": 1.0,    # 1σ above (OTM call / ITM put)
    "M2": -2.0,   # 2σ below
    "P2": 2.0,    # 2σ above
}


def _bs_price(
    spot: float,
    strike: float,
    iv: float,
    T: float,
    option_type: str,
    r: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Black-Scholes price, delta, vega, theta for a European option.
    Returns (price, delta, vega, theta). T in years.
    """
    if T < 1e-8 or iv < 1e-8:
        # At expiry: intrinsic value only
        if option_type == "call":
            intrinsic = max(0.0, spot - strike)
        else:
            intrinsic = max(0.0, strike - spot)
        return intrinsic, 0.0, 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T

    # Standard normal CDF and PDF
    from scipy.stats import norm
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)

    discount = math.exp(-r * T)

    if option_type == "call":
        price = spot * nd1 - strike * discount * nd2
        delta = nd1
    else:
        price = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = nd1 - 1.0

    vega = spot * npd1 * sqrt_T / 100.0  # per 1% IV move
    theta = -(spot * npd1 * iv) / (2.0 * sqrt_T) / 252.0  # per trading day

    return max(price, 0.001), delta, vega, theta


def _parse_contract_key(contract_key: str) -> tuple[str, int, str, str]:
    """Parse 'SPY_7D_ATM_call' -> (underlying, expiry_days, strike_type, option_type)."""
    parts = contract_key.split("_")
    underlying = parts[0]
    expiry_days = int(parts[1].replace("D", ""))
    strike_type = parts[2]
    option_type = parts[3]
    return underlying, expiry_days, strike_type, option_type


def _compute_strike(
    contract_key: str,
    atm_iv: float,
    spot: float,
) -> float:
    """Compute the actual strike price for a contract at the current spot/IV."""
    _, expiry_days, strike_type, _ = _parse_contract_key(contract_key)
    T = expiry_days / 252.0
    sigma_move = spot * atm_iv * math.sqrt(T)
    offset = _STRIKE_OFFSET.get(strike_type, 0.0)
    strike = spot + offset * sigma_move
    return max(strike, spot * 0.5)


def _contract_price(
    contract_key: str,
    atm_iv: float,
    spot: float,
    fixed_strike: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Price a contract using Black-Scholes given current ATM IV and spot.
    Returns (price, delta, vega, theta).

    If fixed_strike is provided, uses that strike (for MTM of held positions).
    Otherwise computes strike from current spot (for new order pricing).
    """
    _, expiry_days, _, option_type = _parse_contract_key(contract_key)
    T = expiry_days / 252.0

    if fixed_strike is not None:
        strike = fixed_strike
    else:
        strike = _compute_strike(contract_key, atm_iv, spot)

    return _bs_price(spot, strike, atm_iv, T, option_type)


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
        "lambda_drawdown_increment": rew.get("lambda_drawdown_increment", 1.0),
        "inaction_penalty": rew.get("inaction_penalty", 0.02),
        "exposure_bonus": rew.get("exposure_bonus", 0.01),
        "max_drawdown_pct": ep.get("max_drawdown_pct", 0.15),
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


def _get_atm_iv(row: dict[str, Any]) -> float:
    """Extract best available ATM IV from a feature_bars row. Falls back through 7d, 30d, VIX/100."""
    for key in ("atm_iv_7d", "atm_iv_30d", "atm_iv_14d"):
        val = row.get(key)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return float(val)
    # Fall back to VIX as a proxy (VIX is in % points, IV is decimal)
    vix = row.get("vix_close")
    if vix is not None and not (isinstance(vix, float) and math.isnan(vix)):
        return float(vix) / 100.0
    return 0.20  # default 20% IV


def _get_spot(row: dict[str, Any], underlying: str) -> float:
    """
    Simulate spot price evolution using equity_return_1d from feature_bars.
    Since we don't have actual spot in feature_bars, we use a reference price
    adjusted by cumulative return. This is called during env init to build
    a spot series.
    """
    return _SPOT_REF.get(underlying, 500.0)


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
        self._positions: dict[str, int] = {}  # contract_key -> qty
        self._position_strikes: dict[str, float] = {}  # contract_key -> fixed entry strike
        self._position_costs: dict[str, float] = {}  # contract_key -> avg cost per contract
        self._peak_equity = initial_cash
        self._daily_pnl_start = initial_cash
        self._day_start_bar = 0
        self._pm_events: list[dict[str, Any]] = []

        # Pre-compute spot path from equity_return_1d for realistic price evolution
        self._spot_path = self._build_spot_path()

    def _build_spot_path(self) -> np.ndarray:
        """Build simulated spot price path from equity_return_1d in feature_bars."""
        base_spot = _SPOT_REF.get(self._underlying, 500.0)
        spots = np.full(len(self._bars) + 1, base_spot, dtype=np.float64)
        for i, bar in enumerate(self._bars):
            ret = bar.get("equity_return_1d")
            if ret is not None and not (isinstance(ret, float) and math.isnan(ret)):
                # equity_return_1d is a daily return; scale to per-bar (~27 bars/day)
                per_bar_ret = float(ret) / 27.0
                spots[i + 1] = spots[i] * (1.0 + per_bar_ret)
            else:
                spots[i + 1] = spots[i]
        return spots

    def _current_spot(self) -> float:
        """Current spot price at bar_idx."""
        idx = min(self._bar_idx, len(self._spot_path) - 1)
        return float(self._spot_path[idx])

    def _current_row(self) -> dict[str, Any]:
        if not self._bars or self._bar_idx >= len(self._bars):
            return {}
        return self._bars[self._bar_idx]

    def _mark_to_market(self) -> float:
        """Value all open positions at current bar's IV/spot prices, using fixed entry strikes."""
        if not self._positions:
            return 0.0
        row = self._current_row()
        atm_iv = _get_atm_iv(row)
        spot = self._current_spot()
        mtm = 0.0
        for ck, qty in self._positions.items():
            if qty == 0:
                continue
            strike = self._position_strikes.get(ck)
            price, _, _, _ = _contract_price(ck, atm_iv, spot, fixed_strike=strike)
            # Option prices are per share, contracts are 100 shares
            mtm += price * qty * 100.0
        return mtm

    def _equity(self) -> float:
        """Equity = cash + mark-to-market value of positions."""
        return self._cash + self._mark_to_market()

    def _portfolio_greeks(self) -> tuple[float, float, float, float]:
        """Compute portfolio greeks from open positions using BS model."""
        net_delta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        net_theta = 0.0
        if not self._positions:
            return net_delta, net_gamma, net_vega, net_theta
        row = self._current_row()
        atm_iv = _get_atm_iv(row)
        spot = self._current_spot()
        for ck, qty in self._positions.items():
            if qty == 0:
                continue
            strike = self._position_strikes.get(ck)
            _, delta, vega, theta = _contract_price(ck, atm_iv, spot, fixed_strike=strike)
            net_delta += delta * qty * 100.0  # per-share delta × 100 shares × qty
            net_vega += vega * qty * 100.0
            net_theta += theta * qty * 100.0
        return net_delta, net_gamma, net_vega, net_theta

    def _apply_fills(self, fills: list[Fill]) -> float:
        """Apply fills: update positions, strikes, and cash. Returns total fees paid."""
        total_fees = 0.0
        for f in fills:
            old_qty = self._positions.get(f.contract_key, 0)
            new_qty = old_qty + f.qty
            # Cash changes: buying costs money, selling generates money
            # price is per-share, multiply by 100 for per-contract
            self._cash -= f.price * f.qty * 100.0
            self._cash -= f.fee
            total_fees += f.fee
            if new_qty == 0:
                self._positions.pop(f.contract_key, None)
                self._position_strikes.pop(f.contract_key, None)
                self._position_costs.pop(f.contract_key, None)
            else:
                self._positions[f.contract_key] = new_qty
                # Store the entry strike (from the order's fixed strike)
                if f.strike > 0 and f.contract_key not in self._position_strikes:
                    self._position_strikes[f.contract_key] = f.strike
        return total_fees

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
        self._position_strikes = {}
        self._position_costs = {}
        self._peak_equity = self._initial_cash
        self._daily_pnl_start = self._initial_cash
        self._day_start_bar = 0
        self._exec = ExecutionSimulator(self._config)
        self._pm_events = [{}] * MAX_PM_EVENTS
        obs = self._get_obs()
        return obs, {"bar_idx": 0}

    def _extract_pm_events(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract PM event data from a feature_bars row into the format _row_to_obs expects."""
        pm_fields = ["p", "logit_p", "delta_p_1h", "delta_p_1d",
                      "momentum", "vol_of_p", "time_to_event", "surprise_z"]
        event: dict[str, Any] = {}
        for field in pm_fields:
            val = row.get(f"pm_{field}")
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                event[field] = float(val)
        # We have one PM event per row; duplicate to fill MAX_PM_EVENTS slots
        if event:
            return [event] + [{}] * (MAX_PM_EVENTS - 1)
        return [{}] * MAX_PM_EVENTS

    def _get_obs(self) -> np.ndarray:
        row = self._current_row()
        nd, ng, nv, nt = self._portfolio_greeks()
        pos_count = sum(1 for q in self._positions.values() if q != 0)
        # Update PM events from current row's prediction market data
        self._pm_events = self._extract_pm_events(row)
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
        """
        Execute one step using TARGET PORTFOLIO model.

        The action defines what the agent WANTS to hold, not incremental orders.
        The env computes the difference between current and target positions,
        then only trades the delta. Action (1,1,x,x) = flatten everything.
        """
        action = np.asarray(action).flatten()
        vega_idx = int(action[0]) if len(action) > 0 else 1
        delta_idx = int(action[1]) if len(action) > 1 else 1
        size_idx = int(action[2]) if len(action) > 2 else 2
        expiry_idx = int(action[3]) if len(action) > 3 else 1
        size_scalar = size_scalar_from_action(size_idx)
        expiry_days = expiry_days_from_action(expiry_idx)
        row = self._current_row()

        # Build TARGET portfolio (what the agent wants to hold)
        target_legs = build_target_positions(
            self._underlying,
            vega_idx,
            delta_idx,
            size_scalar,
            expiry_days,
            base_lots=5,
        )
        target_positions: dict[str, int] = {}
        for ck, qty in target_legs:
            target_positions[ck] = target_positions.get(ck, 0) + qty

        # Compute DELTA orders: difference between target and current positions
        # This lets the agent both open AND close positions
        all_keys = set(self._positions.keys()) | set(target_positions.keys())
        atm_iv = _get_atm_iv(row)
        spot = self._current_spot()
        for ck in all_keys:
            current_qty = self._positions.get(ck, 0)
            target_qty = target_positions.get(ck, 0)
            delta_qty = target_qty - current_qty
            if delta_qty == 0:
                continue

            if delta_qty > 0:
                # OPENING: compute new strike at current spot, price at new strike
                strike = _compute_strike(ck, atm_iv, spot)
                price, _, _, _ = _contract_price(ck, atm_iv, spot, fixed_strike=strike)
            else:
                # CLOSING: use stored entry strike for pricing
                strike = self._position_strikes.get(ck, _compute_strike(ck, atm_iv, spot))
                price, _, _, _ = _contract_price(ck, atm_iv, spot, fixed_strike=strike)

            spread = price * min(0.005 + atm_iv * 0.01, 0.03)  # realistic: 0.5-3%
            vol = 200.0
            self._exec.submit_order(ck, delta_qty, price, spread, vol, self._bar_idx, strike=strike)

        # Measure equity BEFORE advancing (includes current MTM)
        equity_before = self._equity()

        # Advance bar
        self._bar_idx += 1

        # Process fills at new bar's prices
        fills = self._exec.advance_bar(self._bar_idx)
        fees = self._apply_fills(fills)

        # Equity AFTER: cash changed from fills + MTM at new bar's IV/spot
        equity_after = self._equity()
        pnl_step = equity_after - equity_before
        transaction_costs = fees

        has_positions = any(q != 0 for q in self._positions.values())
        reward, self._peak_equity, _ = compute_reward(
            pnl_step,
            transaction_costs,
            self._peak_equity,
            equity_before,
            equity_after,
            self._config.get("lambda_transaction_cost", 1.0),
            self._config.get("lambda_drawdown_increment", 0.1),
            has_positions=has_positions,
            inaction_penalty=self._config.get("inaction_penalty", 0.02),
            exposure_bonus=self._config.get("exposure_bonus", 0.01),
        )
        nd, ng, nv, nt = self._portfolio_greeks()
        premium_at_risk = abs(self._mark_to_market())
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
            reward = -5.0  # Softer penalty (was -100); still bad but not catastrophic
        if check_drawdown_terminate(
            self._peak_equity,
            self._equity(),
            self._config.get("max_drawdown_pct", 0.15),
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
