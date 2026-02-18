#!/usr/bin/env python3
"""Run ablation: 4 variants × PPO/SAC × seeds, save models and results to JSON/CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.ablation import run_ablation, save_ablation_results
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
    out = run_ablation(
        feature_bars=feature_bars,
        algorithms=algorithms,
        seeds=seed_list,
        total_timesteps=timesteps,
        models_dir=Path(models_dir),
        train_pct=0.70,
        val_pct=0.15,
    )
    save_ablation_results(out, json_path=out_json, csv_path=out_csv)
    print(f"Results saved to {out_json} and {out_csv}")
    print("Aggregated (mean ± std):")
    for a in out["aggregated"]:
        print(f"  {a['variant']} {a['algorithm']}: sharpe={a['sharpe_mean']:.4f}±{a['sharpe_std']:.4f}")
    print("P-values vs A:", out["pvalues_vs_A"])


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
    )
