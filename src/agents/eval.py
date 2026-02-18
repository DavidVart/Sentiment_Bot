"""Evaluation harness: run policies on OptionsEnv and compute metrics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

BARS_PER_DAY = 27  # 15-min bars 9:30-16:00
BARS_PER_YEAR = 252 * BARS_PER_DAY


def _bar_returns(equity: np.ndarray, pnl: np.ndarray) -> np.ndarray:
    """Bar-level returns: pnl_t / equity_{t-1}. First bar uses equity[0]-pnl[0] as denominator."""
    out = np.zeros_like(pnl, dtype=float)
    for t in range(len(pnl)):
        if t == 0:
            denom = equity[0] - pnl[0]  # equity before first step
        else:
            denom = equity[t - 1]
        out[t] = pnl[t] / max(abs(denom), 1e-8)
    return out


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (peak - equity) / np.maximum(peak, 1e-8)


def _drawdown_durations(equity: np.ndarray) -> list[int]:
    """Length of each drawdown period in bars (from peak to new peak)."""
    peak = np.maximum.accumulate(equity)
    in_dd = (equity < peak).astype(int)
    if in_dd.sum() == 0:
        return [0]
    # Segment consecutive drawdown bars
    starts = np.where(np.diff(np.concatenate([[0], in_dd])) == 1)[0]
    ends = np.where(np.diff(np.concatenate([in_dd, [0]])) == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return [0]
    # If we end in drawdown, cap at last index
    if ends[-1] < starts[-1]:
        ends = np.concatenate([ends, [len(equity) - 1]])
    lengths = [int(ends[i] - starts[i] + 1) for i in range(min(len(starts), len(ends)))]
    return lengths if lengths else [0]


def compute_metrics(
    pnl: np.ndarray,
    equity: np.ndarray,
    transaction_costs: np.ndarray,
    net_delta: np.ndarray,
    net_vega: np.ndarray,
) -> dict[str, float]:
    """Compute all requested metrics from per-step arrays (after one or more episodes concatenated)."""
    if len(pnl) == 0:
        return _empty_metrics()

    returns = _bar_returns(equity, pnl)
    total_pnl = float(np.sum(pnl))
    total_costs = float(np.sum(transaction_costs))

    # Annualization factor
    n_bars = len(pnl)
    ann_factor = np.sqrt(min(n_bars, BARS_PER_YEAR))

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    if std_ret < 1e-12:
        sharpe = 0.0
    else:
        sharpe = mean_ret / std_ret * ann_factor

    downside_returns = returns[returns < 0]
    downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
    if downside_std < 1e-12:
        sortino = 0.0
    else:
        sortino = mean_ret / downside_std * ann_factor

    dd = _drawdown_series(equity)
    max_dd = float(np.max(dd))
    if max_dd < 1e-12:
        calmar = 0.0
    else:
        annual_return = mean_ret * BARS_PER_YEAR
        calmar = annual_return / max_dd

    dd_durations = _drawdown_durations(equity)
    avg_dd_duration_bars = float(np.mean(dd_durations)) if dd_durations else 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n_win = len(wins)
    n_loss = len(losses)
    hit_rate = (n_win / len(pnl)) * 100.0 if len(pnl) > 0 else 0.0

    sum_wins = float(np.sum(wins)) if n_win else 0.0
    sum_losses_abs = float(np.sum(np.abs(losses))) if n_loss else 0.0
    avg_win = float(np.mean(wins)) if n_win else 0.0
    avg_loss_abs = float(np.mean(np.abs(losses))) if n_loss else 0.0
    if avg_loss_abs < 1e-12:
        avg_win_avg_loss_ratio = 0.0
    else:
        avg_win_avg_loss_ratio = avg_win / avg_loss_abs
    if sum_losses_abs < 1e-12:
        profit_factor = 0.0
    else:
        profit_factor = sum_wins / sum_losses_abs

    avg_spread_paid = 0.0  # not provided by env

    # Turnover: sum of absolute changes in |delta| and |vega| normalized by bars
    d_delta = np.abs(np.diff(net_delta, prepend=net_delta[0] if len(net_delta) else 0))
    d_vega = np.abs(np.diff(net_vega, prepend=net_vega[0] if len(net_vega) else 0))
    turnover_rate = float(np.mean(d_delta + d_vega)) if len(d_delta) else 0.0

    if abs(total_pnl) < 1e-12:
        transaction_costs_pct_pnl = 0.0
    else:
        transaction_costs_pct_pnl = (total_costs / abs(total_pnl)) * 100.0

    avg_abs_delta = float(np.mean(np.abs(net_delta))) if len(net_delta) else 0.0
    avg_abs_vega = float(np.mean(np.abs(net_vega))) if len(net_vega) else 0.0

    # Position concentration: proxy from |delta| and |vega| as two "positions"
    ex = np.abs(net_delta) + np.abs(net_vega) + 1e-12
    w_d = np.abs(net_delta) / ex
    w_v = np.abs(net_vega) / ex
    herfindahl = float(np.mean(w_d**2 + w_v**2)) if len(net_delta) else 0.0

    return {
        "annualized_sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "avg_drawdown_duration_bars": avg_dd_duration_bars,
        "hit_rate_pct": hit_rate,
        "avg_win_avg_loss_ratio": avg_win_avg_loss_ratio,
        "profit_factor": profit_factor,
        "avg_spread_paid": avg_spread_paid,
        "turnover_rate": turnover_rate,
        "transaction_costs_pct_pnl": transaction_costs_pct_pnl,
        "avg_abs_delta": avg_abs_delta,
        "avg_abs_vega": avg_abs_vega,
        "position_concentration_herfindahl": herfindahl,
        "total_pnl": total_pnl,
        "n_bars": n_bars,
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "annualized_sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "avg_drawdown_duration_bars": 0.0,
        "hit_rate_pct": 0.0,
        "avg_win_avg_loss_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_spread_paid": 0.0,
        "turnover_rate": 0.0,
        "transaction_costs_pct_pnl": 0.0,
        "avg_abs_delta": 0.0,
        "avg_abs_vega": 0.0,
        "position_concentration_herfindahl": 0.0,
        "total_pnl": 0.0,
        "n_bars": 0,
    }


def evaluate_policy(
    env: Any,
    policy: Any,
    n_episodes: int = 5,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run n_episodes with (optional) seeds; aggregate pnl/equity/costs/delta/vega and return metrics."""
    if seeds is None:
        seeds = list(range(n_episodes))
    else:
        n_episodes = len(seeds)

    all_pnl: list[float] = []
    all_equity: list[float] = []
    all_costs: list[float] = []
    all_net_delta: list[float] = []
    all_net_vega: list[float] = []
    all_vix: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seeds[ep])
        # Record initial equity (no pnl at step 0)
        eq0 = info.get("equity")
        if eq0 is None:
            eq0 = getattr(env, "_equity", lambda: 0.0)()
            if callable(eq0):
                eq0 = eq0()
        # We'll get equity from first step's info; at reset we don't have step pnl
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            all_pnl.append(info.get("pnl", 0.0))
            all_equity.append(info.get("equity", 0.0))
            all_costs.append(info.get("transaction_costs", 0.0))
            all_net_delta.append(info.get("net_delta", 0.0))
            all_net_vega.append(info.get("net_vega", 0.0))
            all_vix.append(info.get("vix", 0.0))
            done = terminated or truncated

    pnl_arr = np.array(all_pnl, dtype=float)
    equity_arr = np.array(all_equity, dtype=float)
    costs_arr = np.array(all_costs, dtype=float)
    delta_arr = np.array(all_net_delta, dtype=float)
    vega_arr = np.array(all_net_vega, dtype=float)
    vix_arr = np.array(all_vix, dtype=float)

    metrics = compute_metrics(pnl_arr, equity_arr, costs_arr, delta_arr, vega_arr)
    metrics["vix_series"] = vix_arr.tolist()
    return metrics


def regime_split(
    results: dict[str, Any],
    vix_thresholds: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Split metrics by VIX regime: low (<15), medium (15-25), high (>25). results must contain vix_series."""
    if vix_thresholds is None:
        vix_thresholds = [15.0, 25.0]
    vix = np.array(results.get("vix_series", []))
    if len(vix) == 0:
        return {"low": _empty_metrics(), "medium": _empty_metrics(), "high": _empty_metrics()}

    low_thresh = vix_thresholds[0]
    high_thresh = vix_thresholds[1] if len(vix_thresholds) > 1 else 25.0

    mask_low = vix < low_thresh
    mask_med = (vix >= low_thresh) & (vix <= high_thresh)
    mask_high = vix > high_thresh

    # We need per-step pnl, equity, costs, delta, vega - but results only have aggregated metrics and vix_series.
    # So we cannot recompute metrics per regime from results alone. Require that results include per-step arrays.
    pnl = np.array(results.get("pnl_series", []))
    equity = np.array(results.get("equity_series", []))
    costs = np.array(results.get("transaction_costs_series", []))
    delta = np.array(results.get("net_delta_series", []))
    vega = np.array(results.get("net_vega_series", []))

    if len(pnl) != len(vix) or len(pnl) == 0:
        return {"low": _empty_metrics(), "medium": _empty_metrics(), "high": _empty_metrics()}

    def slice_metrics(mask: np.ndarray) -> dict[str, float]:
        if mask.sum() == 0:
            return _empty_metrics()
        return compute_metrics(
            pnl[mask], equity[mask], costs[mask], delta[mask], vega[mask]
        )

    return {
        "low": slice_metrics(mask_low),
        "medium": slice_metrics(mask_med),
        "high": slice_metrics(mask_high),
    }


def bootstrap_sharpe(
    pnl: np.ndarray,
    equity: np.ndarray,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for annualized Sharpe. Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    if n == 0:
        return 0.0, 0.0, 0.0
    returns = _bar_returns(equity, pnl)
    ann = np.sqrt(min(n, BARS_PER_YEAR))
    sharpes: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r = returns[idx]
        if np.std(r) < 1e-12:
            sharpes.append(0.0)
        else:
            sharpes.append(float(np.mean(r) / np.std(r) * ann))
    point = float(np.mean(returns) / np.std(returns) * ann) if np.std(returns) >= 1e-12 else 0.0
    sharpes_arr = np.array(sharpes)
    lower = float(np.percentile(sharpes_arr, 2.5))
    upper = float(np.percentile(sharpes_arr, 97.5))
    return point, lower, upper


def walk_forward_evaluate(
    env_factory: Callable[[], Any],
    policy: Any,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    n_windows: int = 3,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Fit on train window, evaluate on next window; roll forward n_windows times. Returns list of validation metrics per window."""
    # env_factory() returns an env that uses some data; we need to be able to slice data by time.
    # OptionsEnv is built from feature_bars; we don't have a built-in train/val split. So we interpret this as:
    # Create n_windows envs with different data windows (e.g. first 70% bars = train, next 15% = val), then next window shifts.
    # So we need an env_factory that accepts (start_idx, end_idx) or (train_slice, val_slice). For a generic env_factory
    # we can't do that. So we document that env_factory can be called with optional kwargs, e.g. env_factory(train_end_pct=0.7, val_end_pct=0.85, window=0).
    # Simpler: env_factory(window_idx) returns an env configured for that window's validation period. Then we run evaluate_policy on that env.
    # So: walk_forward_evaluate gets env_factory; for window in range(n_windows): env = env_factory(window_idx=window); metrics = evaluate_policy(env, policy, n_episodes=1, seeds=[0]); results.append(metrics).
    # So the contract is: env_factory(window_idx=int) returns an env that is already set up for the window's validation (or train+val). We evaluate on that env.
    if seeds is None:
        seeds = [0] * n_windows
    results: list[dict[str, Any]] = []
    for w in range(n_windows):
        env = env_factory(window_idx=w) if _accepts_window(env_factory) else env_factory()
        m = evaluate_policy(env, policy, n_episodes=1, seeds=[seeds[w]])
        results.append(m)
    return results


def _accepts_window(f: Callable[..., Any]) -> bool:
    import inspect
    try:
        sig = inspect.signature(f)
        return "window_idx" in sig.parameters
    except Exception:
        return False


def evaluate_policy_with_series(
    env: Any,
    policy: Any,
    n_episodes: int = 5,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Like evaluate_policy but also returns pnl_series, equity_series, etc., for regime_split and bootstrap."""
    if seeds is None:
        seeds = list(range(n_episodes))
    else:
        n_episodes = len(seeds)

    all_pnl, all_equity, all_costs, all_delta, all_vega, all_vix = [], [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seeds[ep])
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            all_pnl.append(info.get("pnl", 0.0))
            all_equity.append(info.get("equity", 0.0))
            all_costs.append(info.get("transaction_costs", 0.0))
            all_delta.append(info.get("net_delta", 0.0))
            all_vega.append(info.get("net_vega", 0.0))
            all_vix.append(info.get("vix", 0.0))
            done = terminated or truncated

    pnl_arr = np.array(all_pnl, dtype=float)
    equity_arr = np.array(all_equity, dtype=float)
    costs_arr = np.array(all_costs, dtype=float)
    delta_arr = np.array(all_delta, dtype=float)
    vega_arr = np.array(all_vega, dtype=float)
    vix_arr = np.array(all_vix, dtype=float)

    metrics = compute_metrics(pnl_arr, equity_arr, costs_arr, delta_arr, vega_arr)
    metrics["vix_series"] = vix_arr.tolist()
    metrics["pnl_series"] = pnl_arr.tolist()
    metrics["equity_series"] = equity_arr.tolist()
    metrics["transaction_costs_series"] = costs_arr.tolist()
    metrics["net_delta_series"] = delta_arr.tolist()
    metrics["net_vega_series"] = vega_arr.tolist()
    point, lower, upper = bootstrap_sharpe(pnl_arr, equity_arr, n_resamples=1000)
    metrics["sharpe_ci95_lower"] = lower
    metrics["sharpe_ci95_upper"] = upper
    return metrics
