#!/usr/bin/env python3
"""Generate thesis outputs: tables, plots, case studies from ablation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.analysis import (
    format_ablation_table,
    format_regime_table,
    generate_event_case_study,
    load_ablation_results,
    plot_drawdown_over_time,
    plot_equity_curves,
    plot_exposure_over_time,
    plot_rolling_sharpe,
)
from src.agents.baselines import (
    BuyAndHold,
    DeltaNeutral,
    FixedLongVol,
    RandomPolicy,
    SimpleEventRule,
)
from src.agents.eval import evaluate_policy_with_series
from src.agents.obs_mask_wrapper import ObsMaskWrapper
from src.agents.train_sb3 import split_bars_by_time
from src.envs.options_env import OptionsEnv, load_feature_bars_from_db


def _load_or_fake_bars(use_db: bool, underlying: str = "SPY", limit: int = 2000):
    if use_db:
        return load_feature_bars_from_db(underlying=underlying, limit=limit)
    return _fake_bars(limit)


def _fake_bars(n: int) -> list[dict]:
    return [
        {
            "underlying": "SPY",
            "equity_return_1d": 0.001,
            "realized_vol_5d": 0.15,
            "realized_vol_10d": 0.16,
            "realized_vol_20d": 0.17,
            "realized_vol_60d": 0.18,
            "vix_close": 18.0 + (i % 10),
            "iv_term_slope": -0.01,
            "iv_skew": 0.02,
            "options_gap_flag": False,
            "atm_iv_7d": 0.20,
            "atm_iv_14d": 0.19,
            "atm_iv_30d": 0.18,
            "pm_p": 0.5 + 0.1 * (i % 5) if i > n // 3 else None,
            "sent_news_asset": 0.1,
            "sent_social_asset": 0.0,
            "sent_macro_topic": -0.05,
            "sent_dispersion": 0.02,
            "sent_momentum": 0.01,
            "sent_volume": 5,
            "no_news_flag": False,
        }
        for i in range(n)
    ]


def main(
    ablation_json: str | Path,
    output_dir: str | Path,
    feature_bars_db: str | None = None,
    models_dir: str | Path | None = None,
    underlying: str = "SPY",
    limit_bars: int = 2000,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ablation results
    data = load_ablation_results(ablation_json)

    # (1) Ablation results table
    for algo in ("ppo", "sac"):
        csv_content, md_content = format_ablation_table(data, algorithm=algo)
        if csv_content:
            (output_dir / f"ablation_table_{algo}.csv").write_text(csv_content)
            (output_dir / f"ablation_table_{algo}.md").write_text(md_content)

    # (4) Regime analysis table
    for algo in ("ppo", "sac"):
        csv_r, md_r = format_regime_table(data, algorithm=algo)
        if csv_r:
            (output_dir / f"regime_table_{algo}.csv").write_text(csv_r)
            (output_dir / f"regime_table_{algo}.md").write_text(md_r)

    # (2) Plots: need equity/drawdown/sharpe series from eval
    feature_bars = _load_or_fake_bars(bool(feature_bars_db), underlying=underlying, limit=limit_bars)
    if not feature_bars:
        print("No feature bars; skipping plots and case studies.")
        return
    _, _, test_bars = split_bars_by_time(feature_bars, train_pct=0.70, val_pct=0.15)
    if not test_bars:
        test_bars = feature_bars[-min(500, len(feature_bars)):]

    series_by_label = {}
    pnl_equity_by_label = {}
    vix_series = None
    delta_series_d = None
    vega_series_d = None

    # Baselines
    for name, policy in [
        ("BuyAndHold", BuyAndHold()),
        ("FixedLongVol", FixedLongVol()),
        ("SimpleEventRule", SimpleEventRule()),
        ("DeltaNeutral", DeltaNeutral()),
        ("RandomPolicy", RandomPolicy(seed=42)),
    ]:
        env = OptionsEnv(feature_bars=test_bars)
        m = evaluate_policy_with_series(env, policy, n_episodes=1, seeds=[0])
        eq = m.get("equity_series", [])
        pnl = m.get("pnl_series", [])
        if eq:
            series_by_label[name] = eq
            pnl_equity_by_label[name] = (pnl, eq)
        if vix_series is None and m.get("vix_series"):
            vix_series = m["vix_series"]

    # Variants (need trained models)
    models_dir = Path(models_dir) if models_dir else Path("models")
    try:
        from stable_baselines3 import PPO
        from src.agents.ablation import SB3PolicyAdapter
        for v in ("A", "B", "C", "D"):
            path = models_dir / f"ablation_{v}_ppo_seed0.zip"
            if path.exists():
                model = PPO.load(str(path))
                env = ObsMaskWrapper(OptionsEnv(feature_bars=test_bars), variant=v)
                policy = SB3PolicyAdapter(model, algorithm="ppo")
                m = evaluate_policy_with_series(env, policy, n_episodes=1, seeds=[0])
                eq = m.get("equity_series", [])
                pnl = m.get("pnl_series", [])
                if eq:
                    series_by_label[f"Variant {v}"] = eq
                    pnl_equity_by_label[f"Variant {v}"] = (pnl, eq)
                if v == "D" and m.get("net_delta_series") and m.get("net_vega_series"):
                    delta_series_d = m["net_delta_series"]
                    vega_series_d = m["net_vega_series"]
                if vix_series is None and m.get("vix_series"):
                    vix_series = m["vix_series"]
    except Exception as e:
        print("Could not load SB3 models for variant curves:", e)

    if series_by_label:
        vix_arr = np.array(vix_series) if vix_series else None
        plot_equity_curves(
            series_by_label,
            output_dir / "equity_curves.png",
            vix_series=vix_arr,
            title="Equity curves (variants + baselines)",
        )
        plot_drawdown_over_time(
            series_by_label,
            output_dir / "drawdown_over_time.png",
            title="Drawdown over time",
        )
        plot_rolling_sharpe(
            pnl_equity_by_label,
            output_dir / "rolling_sharpe_30bar.png",
            window=30,
            title="Rolling Sharpe (30-bar)",
        )
    if delta_series_d is not None and vega_series_d is not None:
        plot_exposure_over_time(
            np.array(delta_series_d),
            np.array(vega_series_d),
            output_dir / "exposure_variant_d.png",
            title="Exposure (Variant D)",
        )

    # (3) Event case studies (min 3)
    try:
        from stable_baselines3 import PPO
        model_path = models_dir / "ablation_D_ppo_seed0.zip"
        model = PPO.load(str(model_path)) if model_path.exists() else None
        if model is not None:
            for idx, event_slug in enumerate(["event_1", "event_2", "event_3"]):
                start_bar = max(0, (len(feature_bars) * (idx)) // 3 - 10)
                generate_event_case_study(
                    feature_bars,
                    model,
                    event_slug,
                    window_bars=20,
                    start_bar=start_bar,
                    output_path=output_dir / f"case_study_{event_slug}.png",
                )
        else:
            print("No Variant D model found; skipping case studies.")
    except Exception as e:
        print("Could not generate case studies:", e)

    print("Reports written to", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate thesis tables, plots, and case studies")
    p.add_argument("--ablation-json", required=True, help="Path to ablation_results.json")
    p.add_argument("--output-dir", default="reports", help="Output directory")
    p.add_argument("--feature-bars-db", default=None, help="Load bars from DB (underlying); if not set, use fake bars")
    p.add_argument("--models-dir", default="models", help="Directory with saved ablation models")
    p.add_argument("--underlying", default="SPY")
    p.add_argument("--limit", type=int, default=2000, dest="limit_bars")
    args = p.parse_args()
    main(
        ablation_json=args.ablation_json,
        output_dir=args.output_dir,
        feature_bars_db=args.feature_bars_db,
        models_dir=args.models_dir,
        underlying=args.underlying,
        limit_bars=args.limit_bars,
    )
