"""Ablation runner: train and evaluate 4 observation variants (A=Base, B=+Sentiment, C=+PM, D=Full)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.eval import evaluate_policy_with_series, regime_split
from src.agents.obs_mask_wrapper import ObsMaskWrapper, VARIANT_MASKS
from src.agents.train_sb3 import split_bars_by_time, train_agent
from src.envs.options_env import OptionsEnv

VARIANTS = ("A", "B", "C", "D")  # Base, +Sentiment, +PM, Full
ALGORITHMS = ("ppo", "sac")
METRIC_KEYS = ("annualized_sharpe", "sortino", "calmar", "max_drawdown", "hit_rate_pct", "turnover_rate")


class SB3PolicyAdapter:
    """Expose SB3 model as policy with select_action(obs) for eval harness."""

    def __init__(self, model: Any, algorithm: str = "ppo"):
        self.model = model
        self.algorithm = algorithm.lower()

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Return discrete action (4,) from model.predict(observation, deterministic=True); for SAC, round continuous output to int."""
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.asarray(action).flatten()
        if self.algorithm == "sac":
            action = np.round(np.clip(action, 0.0, 2.0)).astype(np.int64)
        return action


def _run_one(
    variant: str,
    algorithm: str,
    seed: int,
    feature_bars: list[dict],
    total_timesteps: int,
    models_dir: Path,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> dict[str, Any]:
    """Train one model and evaluate on val+test; return metrics dict."""
    train_bars, val_bars, test_bars = split_bars_by_time(feature_bars, train_pct, val_pct)
    if not train_bars or not val_bars:
        return _empty_run_result(variant, algorithm, seed)

    train_env = OptionsEnv(feature_bars=train_bars)
    train_env = ObsMaskWrapper(train_env, variant=variant)
    log_dir = models_dir / "logs" if models_dir else None
    model = train_agent(algorithm, train_env, total_timesteps=total_timesteps, seed=seed, log_dir=log_dir)
    save_path = models_dir / f"ablation_{variant}_{algorithm}_seed{seed}.zip" if models_dir else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))

    # Evaluate on validation then test (same mask)
    val_env = ObsMaskWrapper(OptionsEnv(feature_bars=val_bars), variant=variant)
    test_env = ObsMaskWrapper(OptionsEnv(feature_bars=test_bars), variant=variant)
    policy = SB3PolicyAdapter(model, algorithm=algorithm)

    metrics_val = evaluate_policy_with_series(val_env, policy, n_episodes=1, seeds=[seed + 1000])
    metrics_test = evaluate_policy_with_series(test_env, policy, n_episodes=1, seeds=[seed + 2000])

    # Use test metrics as primary; include val for reference
    out = {
        "variant": variant,
        "algorithm": algorithm,
        "seed": seed,
        "sharpe": metrics_test.get("annualized_sharpe", 0.0),
        "sortino": metrics_test.get("sortino", 0.0),
        "calmar": metrics_test.get("calmar", 0.0),
        "max_drawdown": metrics_test.get("max_drawdown", 0.0),
        "hit_rate": metrics_test.get("hit_rate_pct", 0.0),
        "turnover": metrics_test.get("turnover_rate", 0.0),
        "sharpe_ci95_lower": metrics_test.get("sharpe_ci95_lower"),
        "sharpe_ci95_upper": metrics_test.get("sharpe_ci95_upper"),
        "regime_metrics_test": regime_split(metrics_test, vix_thresholds=[15, 25]),
        "regime_metrics_val": regime_split(metrics_val, vix_thresholds=[15, 25]),
    }
    return out


def _empty_run_result(variant: str, algorithm: str, seed: int) -> dict[str, Any]:
    return {
        "variant": variant,
        "algorithm": algorithm,
        "seed": seed,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "turnover": 0.0,
        "sharpe_ci95_lower": 0.0,
        "sharpe_ci95_upper": 0.0,
        "regime_metrics_test": {},
        "regime_metrics_val": {},
    }


def run_ablation(
    feature_bars: list[dict[str, Any]],
    algorithms: tuple[str, ...] = ("ppo", "sac"),
    seeds: list[int] | None = None,
    total_timesteps: int = 50_000,
    models_dir: str | Path | None = None,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> dict[str, Any]:
    """
    Run ablation: for each variant × algorithm × seed, train and evaluate.
    Returns aggregated results plus p-values vs variant A.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    models_dir = Path(models_dir) if models_dir else Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for algorithm in algorithms:
            if algorithm.lower() not in ALGORITHMS:
                continue
            for seed in seeds:
                row = _run_one(
                    variant=variant,
                    algorithm=algorithm.lower(),
                    seed=seed,
                    feature_bars=feature_bars,
                    total_timesteps=total_timesteps,
                    models_dir=models_dir,
                    train_pct=train_pct,
                    val_pct=val_pct,
                )
                results.append(row)

    # Aggregate mean ± std per (variant, algorithm)
    agg = []
    for v in VARIANTS:
        for algo in algorithms:
            subset = [r for r in results if r["variant"] == v and r["algorithm"] == algo]
            if not subset:
                continue
            agg.append({
                "variant": v,
                "algorithm": algo,
                "sharpe_mean": float(np.mean([r["sharpe"] for r in subset])),
                "sharpe_std": float(np.std([r["sharpe"] for r in subset])),
                "sortino_mean": float(np.mean([r["sortino"] for r in subset])),
                "sortino_std": float(np.std([r["sortino"] for r in subset])),
                "calmar_mean": float(np.mean([r["calmar"] for r in subset])),
                "calmar_std": float(np.std([r["calmar"] for r in subset])),
                "max_drawdown_mean": float(np.mean([r["max_drawdown"] for r in subset])),
                "max_drawdown_std": float(np.std([r["max_drawdown"] for r in subset])),
                "hit_rate_mean": float(np.mean([r["hit_rate"] for r in subset])),
                "hit_rate_std": float(np.std([r["hit_rate"] for r in subset])),
                "turnover_mean": float(np.mean([r["turnover"] for r in subset])),
                "turnover_std": float(np.std([r["turnover"] for r in subset])),
            })

    # P-values vs variant A: paired t-test (by seed) per algorithm
    pvalues_sharpe = {}
    try:
        from scipy import stats
    except ImportError:
        stats = None  # type: ignore[assignment]
    for algo in algorithms:
        base_by_seed = sorted([r for r in results if r["variant"] == "A" and r["algorithm"] == algo], key=lambda x: x["seed"])
        for v in ("B", "C", "D"):
            v_by_seed = sorted([r for r in results if r["variant"] == v and r["algorithm"] == algo], key=lambda x: x["seed"])
            if len(base_by_seed) == len(v_by_seed) and len(base_by_seed) >= 2 and stats is not None:
                base_vals = [r["sharpe"] for r in base_by_seed]
                v_vals = [r["sharpe"] for r in v_by_seed]
                try:
                    _, p = stats.ttest_rel(v_vals, base_vals)
                    pvalues_sharpe[f"pval_sharpe_vs_A_{v}_{algo}"] = float(p)
                except Exception:
                    pvalues_sharpe[f"pval_sharpe_vs_A_{v}_{algo}"] = float("nan")
            else:
                pvalues_sharpe[f"pval_sharpe_vs_A_{v}_{algo}"] = float("nan")

    return {
        "results": results,
        "aggregated": agg,
        "pvalues_vs_A": pvalues_sharpe,
        "config": {
            "algorithms": list(algorithms),
            "seeds": seeds,
            "total_timesteps": total_timesteps,
            "train_pct": train_pct,
            "val_pct": val_pct,
        },
    }


def save_ablation_results(
    ablation_output: dict[str, Any],
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> None:
    """
    Write ablation results to disk. Saves full results and p-values to JSON; per-run metrics to CSV;
    also writes an aggregated CSV (mean ± std per variant/algorithm). Default paths: ablation_results.json, ablation_results.csv.
    """
    import csv
    json_path = Path(json_path) if json_path else Path("ablation_results.json")
    csv_path = Path(csv_path) if csv_path else Path("ablation_results.csv")
    with open(json_path, "w") as f:
        # Drop non-serializable or large nested dicts for JSON
        out = {
            "results": ablation_output["results"],
            "aggregated": ablation_output["aggregated"],
            "pvalues_vs_A": ablation_output["pvalues_vs_A"],
            "config": ablation_output["config"],
        }
        json.dump(out, f, indent=2)
    # CSV: one row per run (variant, algorithm, seed, sharpe, sortino, calmar, max_drawdown, hit_rate, turnover)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "algorithm", "seed", "sharpe", "sortino", "calmar", "max_drawdown", "hit_rate", "turnover"])
        for r in ablation_output["results"]:
            w.writerow([
                r["variant"], r["algorithm"], r["seed"],
                r["sharpe"], r["sortino"], r["calmar"], r["max_drawdown"], r["hit_rate"], r["turnover"],
            ])
    # Optional: aggregated CSV
    agg_path = csv_path.parent / (csv_path.stem + "_aggregated.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "algorithm", "sharpe_mean", "sharpe_std", "sortino_mean", "sortino_std", "calmar_mean", "calmar_std", "max_drawdown_mean", "max_drawdown_std", "hit_rate_mean", "hit_rate_std", "turnover_mean", "turnover_std"])
        for a in ablation_output["aggregated"]:
            w.writerow([
                a["variant"], a["algorithm"],
                a["sharpe_mean"], a["sharpe_std"], a["sortino_mean"], a["sortino_std"],
                a["calmar_mean"], a["calmar_std"], a["max_drawdown_mean"], a["max_drawdown_std"],
                a["hit_rate_mean"], a["hit_rate_std"], a["turnover_mean"], a["turnover_std"],
            ])
