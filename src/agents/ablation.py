"""Ablation runner: train and evaluate 4 observation variants (A=Base, B=+Sentiment, C=+PM, D=Full).

Supports two evaluation modes:
  1. Simple split (default): 70/15/15 temporal train/val/test.
  2. Walk-forward: rolling train/eval windows (configurable days).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.eval import evaluate_policy_with_series, regime_split
from src.agents.obs_mask_wrapper import ObsMaskWrapper, VARIANT_MASKS
from src.agents.train_sb3 import split_bars_by_time, split_bars_walk_forward, train_agent
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
    task_run_id: int | None = None,
    run_index: int = 0,
    total_runs: int = 1,
) -> dict[str, Any]:
    """Train one model and evaluate on val+test; return metrics dict."""
    train_bars, val_bars, test_bars = split_bars_by_time(feature_bars, train_pct, val_pct)
    if not train_bars or not val_bars:
        return _empty_run_result(variant, algorithm, seed)

    train_env = OptionsEnv(feature_bars=train_bars)
    train_env = ObsMaskWrapper(train_env, variant=variant)
    log_dir = models_dir / "logs" if models_dir else None

    # Create progress callback if tracking is enabled
    cb = None
    if task_run_id is not None:
        from src.agents.progress_callback import TrainingProgressCallback
        cb = TrainingProgressCallback(
            task_run_id=task_run_id,
            total_timesteps=total_timesteps,
            variant=variant,
            algorithm=algorithm,
            seed=seed,
            run_index=run_index,
            total_runs=total_runs,
        )

    model = train_agent(algorithm, train_env, total_timesteps=total_timesteps, seed=seed, log_dir=log_dir, callback=cb)
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
    task_run_id: int | None = None,
) -> dict[str, Any]:
    """
    Run ablation: for each variant × algorithm × seed, train and evaluate.
    Returns aggregated results plus p-values vs variant A.
    When task_run_id is provided, progress is written to the task_runs DB table.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    models_dir = Path(models_dir) if models_dir else Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Compute total runs for progress tracking
    valid_algos = [a.lower() for a in algorithms if a.lower() in ALGORITHMS]
    total_runs = len(VARIANTS) * len(valid_algos) * len(seeds)
    run_index = 0

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
                    task_run_id=task_run_id,
                    run_index=run_index,
                    total_runs=total_runs,
                )
                results.append(row)
                run_index += 1

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


def run_baselines(
    feature_bars: list[dict[str, Any]],
    seeds: list[int] | None = None,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Run all 5 baseline strategies through the same evaluation pipeline.
    Each baseline runs on the test partition with multiple seeds.
    Returns list of dicts with same schema as ablation per-run results.
    """
    from src.agents.baselines import (
        BuyAndHold, FixedLongVol, SimpleEventRule, DeltaNeutral, RandomPolicy,
    )

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    _, _, test_bars = split_bars_by_time(feature_bars, train_pct, val_pct)
    if not test_bars:
        return []

    baseline_classes: list[tuple[str, type | object]] = [
        ("buy_and_hold", BuyAndHold),
        ("fixed_long_vol", FixedLongVol),
        ("simple_event_rule", SimpleEventRule),
        ("delta_neutral", DeltaNeutral),
        ("random_policy", RandomPolicy),
    ]

    results: list[dict[str, Any]] = []
    for name, cls in baseline_classes:
        for seed in seeds:
            if name == "random_policy":
                policy = RandomPolicy(seed=seed)
            else:
                policy = cls()  # type: ignore[operator]

            env = OptionsEnv(feature_bars=test_bars)
            metrics = evaluate_policy_with_series(env, policy, n_episodes=1, seeds=[seed])

            results.append({
                "variant": name,
                "algorithm": "baseline",
                "seed": seed,
                "sharpe": metrics.get("annualized_sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "calmar": metrics.get("calmar", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "hit_rate": metrics.get("hit_rate_pct", 0.0),
                "turnover": metrics.get("turnover_rate", 0.0),
                "sharpe_ci95_lower": metrics.get("sharpe_ci95_lower"),
                "sharpe_ci95_upper": metrics.get("sharpe_ci95_upper"),
                "equity_series": metrics.get("equity_series", []),
                "pnl_series": metrics.get("pnl_series", []),
                "net_delta_series": metrics.get("net_delta_series", []),
                "net_vega_series": metrics.get("net_vega_series", []),
            })

    return results


def aggregate_baselines(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate baseline results to mean ± std per baseline name."""
    names = sorted(set(r["variant"] for r in results))
    agg = []
    for name in names:
        subset = [r for r in results if r["variant"] == name]
        agg.append({
            "variant": name,
            "algorithm": "baseline",
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
    return agg


# ---------------------------------------------------------------------------
# Walk-forward ablation
# ---------------------------------------------------------------------------


def _run_one_fold(
    variant: str,
    algorithm: str,
    seed: int,
    fold: dict[str, Any],
    total_timesteps: int,
    models_dir: Path,
    task_run_id: int | None = None,
    run_index: int = 0,
    total_runs: int = 1,
) -> dict[str, Any]:
    """Train on fold's train_bars, evaluate on fold's eval_bars. Returns per-fold metrics."""
    train_bars = fold["train_bars"]
    eval_bars = fold["eval_bars"]
    fold_idx = fold["fold"]

    if not train_bars or not eval_bars:
        out = _empty_run_result(variant, algorithm, seed)
        out["fold"] = fold_idx
        return out

    train_env = ObsMaskWrapper(OptionsEnv(feature_bars=train_bars), variant=variant)

    cb = None
    if task_run_id is not None:
        from src.agents.progress_callback import TrainingProgressCallback
        cb = TrainingProgressCallback(
            task_run_id=task_run_id,
            total_timesteps=total_timesteps,
            variant=variant,
            algorithm=algorithm,
            seed=seed,
            run_index=run_index,
            total_runs=total_runs,
        )

    model = train_agent(algorithm, train_env, total_timesteps=total_timesteps, seed=seed, callback=cb)

    # Save model per fold
    save_path = models_dir / f"wf_{variant}_{algorithm}_seed{seed}_fold{fold_idx}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    eval_env = ObsMaskWrapper(OptionsEnv(feature_bars=eval_bars), variant=variant)
    policy = SB3PolicyAdapter(model, algorithm=algorithm)
    metrics = evaluate_policy_with_series(eval_env, policy, n_episodes=1, seeds=[seed + 3000 + fold_idx])

    return {
        "variant": variant,
        "algorithm": algorithm,
        "seed": seed,
        "fold": fold_idx,
        "train_range": fold["train_range"],
        "eval_range": fold["eval_range"],
        "sharpe": metrics.get("annualized_sharpe", 0.0),
        "sortino": metrics.get("sortino", 0.0),
        "calmar": metrics.get("calmar", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "hit_rate": metrics.get("hit_rate_pct", 0.0),
        "turnover": metrics.get("turnover_rate", 0.0),
        "sharpe_ci95_lower": metrics.get("sharpe_ci95_lower"),
        "sharpe_ci95_upper": metrics.get("sharpe_ci95_upper"),
        "n_bars": metrics.get("n_bars", 0),
        "total_pnl": metrics.get("total_pnl", 0.0),
        "equity_series": metrics.get("equity_series", []),
        "regime_metrics": regime_split(metrics, vix_thresholds=[15, 25]),
    }


def run_ablation_walk_forward(
    feature_bars: list[dict[str, Any]],
    algorithms: tuple[str, ...] = ("ppo", "sac"),
    seeds: list[int] | None = None,
    total_timesteps: int = 50_000,
    models_dir: str | Path | None = None,
    train_days: int = 20,
    eval_days: int = 5,
    task_run_id: int | None = None,
) -> dict[str, Any]:
    """
    Walk-forward ablation: for each fold × variant × algorithm × seed, train and evaluate.

    Defaults: train_days=20, eval_days=5.
    Adjusted for real-time collection period; scales to 60/15 with longer data windows.

    Folds are generated by rolling a (train_days, eval_days) window across feature_bars.
    Returns per-fold results, per-fold aggregation, and cross-fold aggregation.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    models_dir = Path(models_dir) if models_dir else Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    folds = split_bars_walk_forward(feature_bars, train_days=train_days, eval_days=eval_days)
    if not folds:
        return {
            "mode": "walk_forward",
            "results": [],
            "per_fold": [],
            "aggregated": [],
            "pvalues_vs_A": {},
            "config": {
                "algorithms": list(algorithms),
                "seeds": seeds,
                "total_timesteps": total_timesteps,
                "train_days": train_days,
                "eval_days": eval_days,
                "n_folds": 0,
            },
        }

    valid_algos = [a.lower() for a in algorithms if a.lower() in ALGORITHMS]
    total_runs = len(folds) * len(VARIANTS) * len(valid_algos) * len(seeds)
    run_index = 0

    results: list[dict[str, Any]] = []
    for fold in folds:
        for variant in VARIANTS:
            for algorithm in algorithms:
                if algorithm.lower() not in ALGORITHMS:
                    continue
                for seed in seeds:
                    row = _run_one_fold(
                        variant=variant,
                        algorithm=algorithm.lower(),
                        seed=seed,
                        fold=fold,
                        total_timesteps=total_timesteps,
                        models_dir=models_dir,
                        task_run_id=task_run_id,
                        run_index=run_index,
                        total_runs=total_runs,
                    )
                    results.append(row)
                    run_index += 1

    # Per-fold aggregation: mean across seeds for each (fold, variant, algorithm)
    per_fold_agg: list[dict[str, Any]] = []
    for fold in folds:
        fi = fold["fold"]
        for v in VARIANTS:
            for algo in valid_algos:
                subset = [r for r in results
                          if r["fold"] == fi and r["variant"] == v and r["algorithm"] == algo]
                if not subset:
                    continue
                per_fold_agg.append({
                    "fold": fi,
                    "variant": v,
                    "algorithm": algo,
                    "train_range": fold["train_range"],
                    "eval_range": fold["eval_range"],
                    "sharpe_mean": float(np.mean([r["sharpe"] for r in subset])),
                    "sharpe_std": float(np.std([r["sharpe"] for r in subset])),
                    "max_drawdown_mean": float(np.mean([r["max_drawdown"] for r in subset])),
                    "hit_rate_mean": float(np.mean([r["hit_rate"] for r in subset])),
                    "turnover_mean": float(np.mean([r["turnover"] for r in subset])),
                    "total_pnl_mean": float(np.mean([r["total_pnl"] for r in subset])),
                    "n_bars": subset[0].get("n_bars", 0),
                })

    # Cross-fold aggregation: mean ± std across ALL folds and seeds per (variant, algorithm)
    cross_fold_agg: list[dict[str, Any]] = []
    for v in VARIANTS:
        for algo in valid_algos:
            subset = [r for r in results if r["variant"] == v and r["algorithm"] == algo]
            if not subset:
                continue
            cross_fold_agg.append({
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
                "n_folds": len(folds),
                "n_seeds": len(seeds),
            })

    # P-values vs A (paired across folds × seeds)
    pvalues: dict[str, float] = {}
    try:
        from scipy import stats as sp_stats
    except ImportError:
        sp_stats = None  # type: ignore[assignment]
    for algo in valid_algos:
        base = sorted(
            [r for r in results if r["variant"] == "A" and r["algorithm"] == algo],
            key=lambda x: (x["fold"], x["seed"]),
        )
        for v in ("B", "C", "D"):
            other = sorted(
                [r for r in results if r["variant"] == v and r["algorithm"] == algo],
                key=lambda x: (x["fold"], x["seed"]),
            )
            if len(base) == len(other) and len(base) >= 2 and sp_stats is not None:
                try:
                    _, p = sp_stats.ttest_rel(
                        [r["sharpe"] for r in other],
                        [r["sharpe"] for r in base],
                    )
                    pvalues[f"pval_sharpe_vs_A_{v}_{algo}"] = float(p)
                except Exception:
                    pvalues[f"pval_sharpe_vs_A_{v}_{algo}"] = float("nan")
            else:
                pvalues[f"pval_sharpe_vs_A_{v}_{algo}"] = float("nan")

    return {
        "mode": "walk_forward",
        "results": results,
        "per_fold": per_fold_agg,
        "aggregated": cross_fold_agg,
        "pvalues_vs_A": pvalues,
        "config": {
            "algorithms": list(algorithms),
            "seeds": seeds,
            "total_timesteps": total_timesteps,
            "train_days": train_days,
            "eval_days": eval_days,
            "n_folds": len(folds),
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
    # Determine if this is walk-forward (has 'fold' key in results)
    is_wf = ablation_output.get("mode") == "walk_forward"

    # CSV: one row per run
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        if is_wf:
            w.writerow(["variant", "algorithm", "seed", "fold", "sharpe", "sortino", "calmar", "max_drawdown", "hit_rate", "turnover", "total_pnl", "n_bars"])
        else:
            w.writerow(["variant", "algorithm", "seed", "sharpe", "sortino", "calmar", "max_drawdown", "hit_rate", "turnover"])
        for r in ablation_output["results"]:
            if is_wf:
                w.writerow([
                    r["variant"], r["algorithm"], r["seed"], r.get("fold", ""),
                    r["sharpe"], r["sortino"], r["calmar"], r["max_drawdown"],
                    r["hit_rate"], r["turnover"], r.get("total_pnl", 0.0), r.get("n_bars", 0),
                ])
            else:
                w.writerow([
                    r["variant"], r["algorithm"], r["seed"],
                    r["sharpe"], r["sortino"], r["calmar"], r["max_drawdown"], r["hit_rate"], r["turnover"],
                ])

    # Aggregated CSV
    agg_path = csv_path.parent / (csv_path.stem + "_aggregated.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "algorithm", "sharpe_mean", "sharpe_std", "sortino_mean", "sortino_std", "calmar_mean", "calmar_std", "max_drawdown_mean", "max_drawdown_std", "hit_rate_mean", "hit_rate_std", "turnover_mean", "turnover_std"])
        for a in ablation_output["aggregated"]:
            w.writerow([
                a["variant"], a["algorithm"],
                a["sharpe_mean"], a["sharpe_std"],
                a.get("sortino_mean", 0.0), a.get("sortino_std", 0.0),
                a.get("calmar_mean", 0.0), a.get("calmar_std", 0.0),
                a["max_drawdown_mean"], a.get("max_drawdown_std", 0.0),
                a["hit_rate_mean"], a.get("hit_rate_std", 0.0),
                a["turnover_mean"], a.get("turnover_std", 0.0),
            ])

    # Per-fold CSV (walk-forward only)
    if is_wf and ablation_output.get("per_fold"):
        fold_path = csv_path.parent / (csv_path.stem + "_per_fold.csv")
        with open(fold_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "variant", "algorithm", "sharpe_mean", "sharpe_std",
                         "max_drawdown_mean", "hit_rate_mean", "turnover_mean",
                         "total_pnl_mean", "n_bars", "train_range", "eval_range"])
            for pf in ablation_output["per_fold"]:
                w.writerow([
                    pf["fold"], pf["variant"], pf["algorithm"],
                    pf["sharpe_mean"], pf["sharpe_std"],
                    pf["max_drawdown_mean"], pf["hit_rate_mean"], pf["turnover_mean"],
                    pf["total_pnl_mean"], pf["n_bars"],
                    str(pf["train_range"]), str(pf["eval_range"]),
                ])
