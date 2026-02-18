#!/usr/bin/env python3
"""Run all five baselines on the same OptionsEnv data and output a comparison table + JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.baselines import (
    BuyAndHold,
    DeltaNeutral,
    FixedLongVol,
    RandomPolicy,
    SimpleEventRule,
)
from src.agents.eval import (
    evaluate_policy_with_series,
    regime_split,
)
from src.envs.options_env import OptionsEnv, load_feature_bars_from_db


def make_env(feature_bars: list, underlying: str = "SPY"):
    return OptionsEnv(feature_bars=feature_bars, underlying=underlying)


def main(
    underlying: str = "SPY",
    limit: int = 2000,
    n_episodes: int = 3,
    seeds: list[int] | None = None,
    out_json: str | None = None,
) -> dict:
    if seeds is None:
        seeds = list(range(n_episodes))

    feature_bars = load_feature_bars_from_db(underlying=underlying, limit=limit)
    if not feature_bars:
        print("No feature bars loaded. Exiting.")
        return {}

    env = make_env(feature_bars, underlying)
    policies = {
        "BuyAndHold": BuyAndHold(),
        "FixedLongVol": FixedLongVol(),
        "SimpleEventRule": SimpleEventRule(),
        "DeltaNeutral": DeltaNeutral(),
        "RandomPolicy": RandomPolicy(seed=42),
    }

    results: dict[str, dict] = {}
    for name, policy in policies.items():
        env = make_env(feature_bars, underlying)
        metrics = evaluate_policy_with_series(env, policy, n_episodes=n_episodes, seeds=seeds)
        # Drop series for JSON (keep only scalars + regime split)
        regime = regime_split(metrics, vix_thresholds=[15, 25])
        results[name] = {
            k: v for k, v in metrics.items()
            if k not in ("vix_series", "pnl_series", "equity_series", "transaction_costs_series", "net_delta_series", "net_vega_series")
        }
        results[name]["regime_metrics"] = regime

    # Print table
    key_metrics = [
        "annualized_sharpe", "sortino", "calmar", "max_drawdown",
        "hit_rate_pct", "profit_factor", "total_pnl", "transaction_costs_pct_pnl",
        "avg_abs_delta", "avg_abs_vega", "sharpe_ci95_lower", "sharpe_ci95_upper",
    ]
    col_w = 18
    header = "Policy".ljust(20) + "".join(m.ljust(col_w) for m in key_metrics)
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        row = name.ljust(20)
        for m in key_metrics:
            v = data.get(m)
            if v is None:
                row += "-".ljust(col_w)
            elif isinstance(v, float):
                row += f"{v:.4f}".ljust(col_w)
            else:
                row += str(v).ljust(col_w)
        print(row)

    out = {"policies": list(results.keys()), "metrics": results}
    if out_json:
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {out_json}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run baseline policies and compare metrics")
    p.add_argument("--underlying", default="SPY")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--out", dest="out_json", default=None, help="Output JSON path")
    args = p.parse_args()
    main(
        underlying=args.underlying,
        limit=args.limit,
        n_episodes=args.n_episodes,
        out_json=args.out_json,
    )
