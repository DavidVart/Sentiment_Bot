"""Thesis outputs: ablation tables, plots, event case studies, regime analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Optional matplotlib for plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

VARIANT_LABELS = {"A": "Base", "B": "+Sentiment", "C": "+PM", "D": "Full"}
VIX_THRESHOLDS = (15.0, 25.0)


def load_ablation_results(path: str | Path) -> dict[str, Any]:
    """Load ablation JSON; return dict with results, aggregated, pvalues_vs_A, config."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def format_ablation_table(
    ablation_data: dict[str, Any],
    algorithm: str = "ppo",
) -> tuple[str, str]:
    """
    Build publication-ready ablation table: Variant A–D rows, Sharpe/Sortino/Calmar/max_dd/hit_rate/turnover (mean ± std), p-values vs A.
    Returns (csv_content, markdown_content).
    """
    agg = [a for a in ablation_data.get("aggregated", []) if a.get("algorithm") == algorithm]
    pvals = ablation_data.get("pvalues_vs_A", {})
    if not agg:
        return "", ""

    rows = []
    for v in ("A", "B", "C", "D"):
        a = next((x for x in agg if x["variant"] == v), None)
        if not a:
            continue
        pkey = f"pval_sharpe_vs_A_{v}_{algorithm}"
        pval = pvals.get(pkey, float("nan"))
        pval_str = f"{pval:.4f}" if not np.isnan(pval) else "—"
        if v == "A":
            pval_str = "—"
        rows.append({
            "Variant": VARIANT_LABELS.get(v, v),
            "Sharpe": f"{a['sharpe_mean']:.4f} ± {a['sharpe_std']:.4f}",
            "Sortino": f"{a['sortino_mean']:.4f} ± {a['sortino_std']:.4f}",
            "Calmar": f"{a['calmar_mean']:.4f} ± {a['calmar_std']:.4f}",
            "Max DD": f"{a['max_drawdown_mean']:.4f} ± {a['max_drawdown_std']:.4f}",
            "Hit rate (%)": f"{a['hit_rate_mean']:.2f} ± {a['hit_rate_std']:.2f}",
            "Turnover": f"{a['turnover_mean']:.4f} ± {a['turnover_std']:.4f}",
            "p (vs A)": pval_str,
        })

    cols = ["Variant", "Sharpe", "Sortino", "Calmar", "Max DD", "Hit rate (%)", "Turnover", "p (vs A)"]
    # CSV
    csv_lines = [",".join(cols)]
    for r in rows:
        csv_lines.append(",".join(str(r[c]) for c in cols))
    csv_content = "\n".join(csv_lines)

    # Markdown
    md_lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md_content = "\n".join(md_lines)
    return csv_content, md_content


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (peak - equity) / np.maximum(peak, 1e-8)


def _rolling_sharpe(pnl: np.ndarray, equity: np.ndarray, window: int) -> np.ndarray:
    n = len(pnl)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        eq = equity[t - window + 1 : t + 1]
        p = pnl[t - window + 1 : t + 1]
        ret = np.zeros(window)
        for i in range(window):
            denom = eq[i - 1] if i > 0 else (eq[0] - p[0])
            ret[i] = p[i] / max(abs(denom), 1e-8)
        if np.std(ret) < 1e-12:
            out[t] = 0.0
        else:
            out[t] = np.mean(ret) / np.std(ret) * np.sqrt(252 * 27)
    return out


def plot_equity_curves(
    series_dict: dict[str, np.ndarray],
    output_path: str | Path,
    vix_series: np.ndarray | None = None,
    title: str = "Equity curves",
) -> None:
    """Overlay equity curves; optional VIX regime shading (low <15, medium 15–25, high >25)."""
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    max_len = 0
    for label, eq in series_dict.items():
        eq = np.asarray(eq).flatten()
        if len(eq) == 0:
            continue
        max_len = max(max_len, len(eq))
        ax.plot(np.arange(len(eq)), eq, label=label, alpha=0.8)
    if vix_series is not None:
        vix = np.asarray(vix_series).flatten()
        n = min(len(vix), max_len)
        for i in range(n - 1):
            v = vix[i]
            if v < VIX_THRESHOLDS[0]:
                ax.axvspan(i, i + 1, alpha=0.1, color="green")
            elif v > VIX_THRESHOLDS[1]:
                ax.axvspan(i, i + 1, alpha=0.1, color="red")
            else:
                ax.axvspan(i, i + 1, alpha=0.05, color="gray")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown_over_time(
    series_dict: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Drawdown over time",
) -> None:
    """Plot drawdown (0 to 1) over time for each series."""
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, eq in series_dict.items():
        eq = np.asarray(eq).flatten()
        if len(eq) == 0:
            continue
        dd = _drawdown_series(eq)
        ax.plot(np.arange(len(dd)), dd, label=label, alpha=0.8)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(
    series_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    window: int = 30,
    title: str = "Rolling Sharpe (30-bar)",
) -> None:
    """Plot rolling annualized Sharpe ratio over time. series_dict maps label -> (pnl_series, equity_series). Saves figure to output_path."""
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (pnl, eq) in series_dict.items():
        pnl = np.asarray(pnl).flatten()
        eq = np.asarray(eq).flatten()
        if len(pnl) < window or len(eq) < window:
            continue
        rs = _rolling_sharpe(pnl, eq, window)
        valid = ~np.isnan(rs)
        if np.any(valid):
            ax.plot(np.arange(len(rs)), rs, label=label, alpha=0.8)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Rolling Sharpe")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_exposure_over_time(
    delta_series: np.ndarray,
    vega_series: np.ndarray,
    output_path: str | Path,
    title: str = "Exposure (Variant D)",
) -> None:
    """Net delta and net vega over time (e.g. full model)."""
    if not _HAS_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    t = np.arange(len(delta_series))
    ax1.plot(t, np.asarray(delta_series).flatten(), color="C0", label="Net delta")
    ax1.set_ylabel("Net delta")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax2.plot(t, np.asarray(vega_series).flatten(), color="C1", label="Net vega")
    ax2.set_ylabel("Net vega")
    ax2.set_xlabel("Bar")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_regime_table(ablation_data: dict[str, Any], algorithm: str = "ppo") -> tuple[str, str]:
    """
    All metrics by VIX regime (low <15, medium 15–25, high >25) per variant. Uses first result per variant that has regime_metrics_test.
    Returns (csv_content, markdown_content).
    """
    results = [r for r in ablation_data.get("results", []) if r.get("algorithm") == algorithm]
    regime_rows = []
    for v in ("A", "B", "C", "D"):
        r = next((x for x in results if x["variant"] == v), None)
        if not r:
            continue
        reg = r.get("regime_metrics_test") or {}
        for regime_name in ("low", "medium", "high"):
            m = reg.get(regime_name) or {}
            regime_rows.append({
                "Variant": VARIANT_LABELS.get(v, v),
                "Regime": regime_name,
                "Sharpe": m.get("annualized_sharpe", 0.0),
                "Sortino": m.get("sortino", 0.0),
                "Calmar": m.get("calmar", 0.0),
                "Max DD": m.get("max_drawdown", 0.0),
                "Hit rate (%)": m.get("hit_rate_pct", 0.0),
                "Turnover": m.get("turnover_rate", 0.0),
                "n_bars": m.get("n_bars", 0),
            })

    if not regime_rows:
        return "", ""

    cols = ["Variant", "Regime", "Sharpe", "Sortino", "Calmar", "Max DD", "Hit rate (%)", "Turnover", "n_bars"]
    csv_lines = [",".join(cols)]
    for row in regime_rows:
        csv_lines.append(",".join(str(row[c]) for c in cols))
    csv_content = "\n".join(csv_lines)

    md_lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in regime_rows:
        md_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    md_content = "\n".join(md_lines)
    return csv_content, md_content


def generate_event_case_study(
    feature_bars: list[dict[str, Any]],
    model: Any,
    event_slug: str,
    window_bars: int = 20,
    start_bar: int | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """
    Multi-panel figure for one prediction-market event: PM probability, IV move, agent actions, P&L decomposition.
    model must have predict(obs) or select_action(obs). If start_bar is None, use first bar with non-null pm_p or center at len(bars)//2.
    """
    if not _HAS_MPL:
        return None
    bars = feature_bars
    n = len(bars)
    if n == 0:
        return None
    if start_bar is None:
        for i, b in enumerate(bars):
            if b.get("pm_p") is not None:
                start_bar = max(0, i - window_bars // 2)
                break
        if start_bar is None:
            start_bar = max(0, n // 2 - window_bars // 2)
    end_bar = min(start_bar + window_bars, n)
    window = bars[start_bar:end_bar]

    # PM probability trajectory
    pm_p = [float(b.get("pm_p") or 0.0) for b in window]
    # IV (e.g. atm_iv_30d)
    iv = [float(b.get("atm_iv_30d") or 0.0) for b in window]
    # Run agent and collect actions, pnl, delta, vega
    from src.envs.options_env import OptionsEnv
    from src.agents.ablation import SB3PolicyAdapter
    from src.agents.obs_mask_wrapper import ObsMaskWrapper

    env = ObsMaskWrapper(OptionsEnv(feature_bars=window), variant="D")
    policy = SB3PolicyAdapter(model, algorithm="ppo") if hasattr(model, "predict") else model
    obs, _ = env.reset(seed=0)
    actions_list = []
    pnl_list = []
    delta_list = []
    vega_list = []
    equity_list = []
    for _ in range(len(window) - 1):
        action = policy.select_action(obs) if hasattr(policy, "select_action") else model.predict(obs, deterministic=True)[0]
        action = np.asarray(action).flatten()
        if action.dtype.kind == "f":
            action = np.round(np.clip(action, 0, 2)).astype(np.int64)
        actions_list.append(action.copy())
        obs, _, term, trunc, info = env.step(action)
        pnl_list.append(info.get("pnl", 0.0))
        delta_list.append(info.get("net_delta", 0.0))
        vega_list.append(info.get("net_vega", 0.0))
        equity_list.append(info.get("equity", 0.0))
        if term or trunc:
            break

    # P&L decomposition: delta_pnl ~ net_delta * return; vega_pnl ~ net_vega * iv_change; residual
    returns = [float(b.get("equity_return_1d") or 0.0) / max(len(window), 1) for b in window[: len(pnl_list) + 1]]
    if len(returns) > len(pnl_list):
        returns = returns[: len(pnl_list)]
    elif len(returns) < len(pnl_list):
        returns = returns + [0.0] * (len(pnl_list) - len(returns))
    iv_changes = [0.0] + [iv[i + 1] - iv[i] for i in range(len(iv) - 1)]
    if len(iv_changes) > len(pnl_list):
        iv_changes = iv_changes[: len(pnl_list)]

    delta_pnl = []
    vega_pnl = []
    for i in range(len(pnl_list)):
        d = delta_list[i] if i < len(delta_list) else 0.0
        v = vega_list[i] if i < len(vega_list) else 0.0
        delta_pnl.append(d * (returns[i] if i < len(returns) else 0.0) * 100)
        vega_pnl.append(v * (iv_changes[i] if i < len(iv_changes) else 0.0) * 10)
    theta_pnl = [0.0] * len(pnl_list)
    residual = [float(pnl_list[i]) - delta_pnl[i] - vega_pnl[i] - theta_pnl[i] for i in range(len(pnl_list))]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    x = np.arange(len(window))

    axes[0].plot(x, pm_p, "o-", color="C0", label="PM probability")
    axes[0].set_ylabel("P (event)")
    axes[0].set_title(f"Event: {event_slug}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, iv, "s-", color="C1", label="ATM IV 30d")
    axes[1].set_ylabel("IV")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if actions_list:
        act_arr = np.array(actions_list)
        axes[2].plot(np.arange(len(act_arr)), act_arr[:, 0], label="vega bucket", alpha=0.8)
        axes[2].plot(np.arange(len(act_arr)), act_arr[:, 1], label="delta bucket", alpha=0.8)
        axes[2].set_ylabel("Action")
        axes[2].legend(loc="best", fontsize=8)
        axes[2].grid(True, alpha=0.3)

    axes[3].plot(np.arange(len(pnl_list)), pnl_list, label="Total P&L", color="black")
    axes[3].plot(np.arange(len(pnl_list)), delta_pnl, label="Delta", alpha=0.8)
    axes[3].plot(np.arange(len(pnl_list)), vega_pnl, label="Vega", alpha=0.8)
    axes[3].plot(np.arange(len(pnl_list)), residual, label="Residual", alpha=0.8)
    axes[3].set_ylabel("P&L")
    axes[3].set_xlabel("Bar")
    axes[3].legend(loc="best", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"Case study: {event_slug}", y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return None
    return fig
