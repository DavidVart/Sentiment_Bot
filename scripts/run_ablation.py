#!/usr/bin/env python3
"""Run ablation: 4 variants × PPO/SAC × seeds, save models and results to JSON/CSV.

Supports two modes:
  --walk-forward    Use rolling train/eval windows (walk-forward validation)
  (default)         Simple 70/15/15 temporal split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.agents.ablation import (
    run_ablation,
    run_ablation_walk_forward,
    save_ablation_results,
)
from src.envs.options_env import load_feature_bars_from_db


def main(
    algorithm: str = "both",
    seeds: int = 5,
    timesteps: int = 50_000,
    underlying: str = "SPY",
    limit: int = 5000,
    models_dir: str | Path = "models",
    out_json: str | Path = "ablation_results.json",
    out_csv: str | Path = "ablation_results.csv",
    task_run_id: int | None = None,
    walk_forward: bool = False,
    train_days: int = 20,
    eval_days: int = 5,
) -> None:
    algorithms = ("ppo", "sac") if algorithm == "both" else (algorithm.lower(),)
    if algorithm != "both" and algorithm.lower() not in ("ppo", "sac"):
        print("--algorithm must be ppo, sac, or both")
        sys.exit(1)
    seed_list = list(range(seeds))
    feature_bars = load_feature_bars_from_db(underlying=underlying, limit=limit)
    if not feature_bars:
        print("No feature bars loaded. Exiting.")
        sys.exit(1)

    if walk_forward:
        print(f"Walk-forward mode: train_days={train_days}, eval_days={eval_days}")
        out = run_ablation_walk_forward(
            feature_bars=feature_bars,
            algorithms=algorithms,
            seeds=seed_list,
            total_timesteps=timesteps,
            models_dir=Path(models_dir),
            train_days=train_days,
            eval_days=eval_days,
            task_run_id=task_run_id,
        )
        n_folds = out["config"]["n_folds"]
        print(f"Walk-forward completed: {n_folds} folds")
    else:
        out = run_ablation(
            feature_bars=feature_bars,
            algorithms=algorithms,
            seeds=seed_list,
            total_timesteps=timesteps,
            models_dir=Path(models_dir),
            train_pct=0.70,
            val_pct=0.15,
            task_run_id=task_run_id,
        )

    save_ablation_results(out, json_path=out_json, csv_path=out_csv)
    print(f"Results saved to {out_json} and {out_csv}")
    print("Aggregated (mean ± std):")
    for a in out["aggregated"]:
        print(f"  {a['variant']} {a['algorithm']}: sharpe={a['sharpe_mean']:.4f}±{a['sharpe_std']:.4f}")
    print("P-values vs A:", out["pvalues_vs_A"])

    if walk_forward and out.get("per_fold"):
        print(f"\nPer-fold breakdown ({n_folds} folds):")
        for pf in out["per_fold"]:
            print(f"  Fold {pf['fold']} {pf['variant']} {pf['algorithm']}: "
                  f"sharpe={pf['sharpe_mean']:.4f} dd={pf['max_drawdown_mean']:.4f} "
                  f"pnl={pf['total_pnl_mean']:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run ablation (4 variants × algorithm × seeds)")
    p.add_argument("--algorithm", choices=("ppo", "sac", "both"), default="both")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--underlying", default="SPY")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out-json", default="ablation_results.json")
    p.add_argument("--out-csv", default="ablation_results.csv")
    p.add_argument("--task-run-id", type=int, default=None, help="Task run ID for progress tracking")
    p.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation instead of simple split")
    # Adjusted for real-time collection period; scales to 60/15 with longer data windows.
    p.add_argument("--train-days", type=int, default=20, help="Training window size in trading days (walk-forward)")
    p.add_argument("--eval-days", type=int, default=5, help="Evaluation window size in trading days (walk-forward)")
    args = p.parse_args()
    main(
        algorithm=args.algorithm,
        seeds=args.seeds,
        timesteps=args.timesteps,
        underlying=args.underlying,
        limit=args.limit,
        models_dir=args.models_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        task_run_id=args.task_run_id,
        walk_forward=args.walk_forward,
        train_days=args.train_days,
        eval_days=args.eval_days,
    )
