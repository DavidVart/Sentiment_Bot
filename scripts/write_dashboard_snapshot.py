#!/usr/bin/env python3
"""
Write precomputed dashboard data to DB so the web API can serve without running the RL model.
Run after the pipeline (e.g. after steps 2–10 and optionally 12). Reads pipeline_runs.log,
ablation_results.json, runs evaluation for configured underlyings, and writes to
dashboard_pipeline_log, dashboard_ablation, dashboard_evaluation, dashboard_trade_impact.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def _serialize_ts(ts) -> str | None:
    if ts is None:
        return None
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _headlines_for_window(cur, ts_start, ts_end, limit: int = 5) -> list[dict]:
    """Fetch sentiment_scored rows in [ts_start, ts_end]; return list of {text, source, compound, model, ts}."""
    if ts_start is None and ts_end is None:
        return []
    try:
        cur.execute(
            """
            SELECT ts, source, left(text, 400) AS snippet, sentiment_compound, sentiment_model
            FROM sentiment_scored
            WHERE ts >= %s AND ts <= %s
            ORDER BY ts DESC
            LIMIT %s
            """,
            (ts_start, ts_end, limit),
        )
        rows = cur.fetchall()
    except Exception:
        return []
    out = []
    for r in rows:
        ts_val, source, snippet, compound, model = r
        out.append({
            "ts": _serialize_ts(ts_val),
            "source": source or "",
            "text": (snippet or "").strip(),
            "compound": float(compound) if compound is not None else 0.0,
            "model": model or "",
        })
    return out


def main() -> None:
    from src.db import get_connection, apply_migrations

    apply_migrations()

    underlyings_str = os.environ.get("DASHBOARD_SNAPSHOT_UNDERLYINGS", "SPY")
    underlyings = [s.strip() for s in underlyings_str.split(",") if s.strip()] or ["SPY"]
    limit_bars = int(os.environ.get("DASHBOARD_SNAPSHOT_LIMIT", "2000"))
    models_dir = Path(os.environ.get("DASHBOARD_MODELS_DIR", str(ROOT / "models")))
    ablation_path = ROOT / "ablation_results.json"
    log_path = ROOT / "pipeline_runs.log"

    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. dashboard_pipeline_log
            last_line = ""
            if log_path.exists():
                lines = log_path.read_text().strip().splitlines()
                last_line = lines[-1] if lines else ""
            cur.execute(
                "SELECT sentiment_model, count(*) FROM sentiment_scored GROUP BY sentiment_model"
            )
            rows = cur.fetchall()
            sentiment_summary = ", ".join(f"{m or 'unknown'}: {c}" for m, c in rows) if rows else "—"
            cur.execute(
                """
                INSERT INTO dashboard_pipeline_log (id, updated_at, last_line, sentiment_summary)
                VALUES (1, NOW(), %s, %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = NOW(), last_line = EXCLUDED.last_line, sentiment_summary = EXCLUDED.sentiment_summary
                """,
                (last_line, sentiment_summary),
            )

            # 2. dashboard_ablation
            if ablation_path.exists():
                try:
                    data = json.loads(ablation_path.read_text())
                    agg = data.get("aggregated", [])
                    cur.execute(
                        """
                        INSERT INTO dashboard_ablation (id, updated_at, aggregated_json)
                        VALUES (1, NOW(), %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET updated_at = NOW(), aggregated_json = EXCLUDED.aggregated_json
                        """,
                        (json.dumps(agg),),
                    )
                except Exception as e:
                    print(f"Warning: could not write dashboard_ablation: {e}", file=sys.stderr)

            # 3 & 4. dashboard_evaluation and dashboard_trade_impact per underlying
            model_path = models_dir / "ablation_D_ppo_seed0.zip"
            if not model_path.exists():
                print("Warning: model not found, skipping evaluation snapshot", file=sys.stderr)
            else:
                from src.envs.options_env import load_feature_bars_from_db, OptionsEnv
                from src.agents.eval import evaluate_policy_with_series
                from src.agents.obs_mask_wrapper import ObsMaskWrapper
                from src.agents.ablation import SB3PolicyAdapter
                from src.agents.baselines import BuyAndHold
                from stable_baselines3 import PPO

                for underlying in underlyings:
                    try:
                        bars = load_feature_bars_from_db(underlying=underlying, limit=limit_bars)
                    except Exception as e:
                        print(f"Warning: could not load feature_bars for {underlying}: {e}", file=sys.stderr)
                        continue
                    if not bars or len(bars) < 2:
                        continue

                    bars_df = None
                    try:
                        import pandas as pd
                        bars_df = pd.DataFrame(bars)
                        if "ts" in bars_df.columns:
                            bars_df["ts"] = pd.to_datetime(bars_df["ts"])
                        bars = bars_df.to_dict("records")
                    except Exception:
                        pass

                    env = OptionsEnv(feature_bars=bars, underlying=underlying)
                    policy_bh = BuyAndHold()
                    metrics_bh = evaluate_policy_with_series(env, policy_bh, n_episodes=1, seeds=[0])
                    metrics_d = None
                    try:
                        model = PPO.load(str(model_path))
                        env2 = ObsMaskWrapper(OptionsEnv(feature_bars=bars, underlying=underlying), variant="D")
                        policy_d = SB3PolicyAdapter(model, algorithm="ppo")
                        metrics_d = evaluate_policy_with_series(env2, policy_d, n_episodes=1, seeds=[0])
                    except Exception as e:
                        print(f"Warning: Variant D eval failed for {underlying}: {e}", file=sys.stderr)

                    eq_bh = metrics_bh.get("equity_series") or []
                    eq_d = (metrics_d.get("equity_series") or []) if metrics_d else []
                    n_ts = min(len(bars) - 1, len(eq_bh)) if eq_bh else 0
                    if n_ts <= 0:
                        n_ts = min(len(bars) - 1, len(eq_d)) if eq_d else 0
                    ts_index = [_serialize_ts(bars[i].get("ts")) for i in range(1, 1 + n_ts) if i < len(bars)]
                    n_ts = len(ts_index)
                    equity_bh = (eq_bh[:n_ts] + [eq_bh[-1]] * max(0, n_ts - len(eq_bh)))[:n_ts] if eq_bh else [0.0] * n_ts
                    equity_d = (eq_d[:n_ts] + [eq_d[-1]] * max(0, n_ts - len(eq_d)))[:n_ts] if eq_d else [0.0] * n_ts

                    metrics_bh_clean = {k: v for k, v in (metrics_bh or {}).items() if k not in ("equity_series", "pnl_series", "net_delta_series", "net_vega_series", "vix_series", "transaction_costs_series")}
                    metrics_d_clean = {k: v for k, v in (metrics_d or {}).items() if k not in ("equity_series", "pnl_series", "net_delta_series", "net_vega_series", "vix_series", "transaction_costs_series")} if metrics_d else {}

                    cur.execute(
                        """
                        INSERT INTO dashboard_evaluation (underlying, updated_at, ts_index, equity_bh, equity_d, metrics_bh, metrics_d, exposure_delta, exposure_vega)
                        VALUES (%s, NOW(), %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                        ON CONFLICT (underlying) DO UPDATE SET
                            updated_at = NOW(), ts_index = EXCLUDED.ts_index, equity_bh = EXCLUDED.equity_bh,
                            equity_d = EXCLUDED.equity_d, metrics_bh = EXCLUDED.metrics_bh, metrics_d = EXCLUDED.metrics_d,
                            exposure_delta = EXCLUDED.exposure_delta, exposure_vega = EXCLUDED.exposure_vega
                        """,
                        (
                            underlying,
                            json.dumps(ts_index),
                            json.dumps(equity_bh),
                            json.dumps(equity_d),
                            json.dumps(metrics_bh_clean),
                            json.dumps(metrics_d_clean),
                            json.dumps(metrics_d.get("net_delta_series", []) if metrics_d else []),
                            json.dumps(metrics_d.get("net_vega_series", []) if metrics_d else []),
                        ),
                    )

                    # Headlines in full bar range (for attaching to each action bar)
                    import pandas as pd
                    bar_tss = [b.get("ts") for b in bars if b.get("ts")]
                    try:
                        bar_tss_parsed = pd.to_datetime(bar_tss, utc=True)
                        t_min, t_max = bar_tss_parsed.min(), bar_tss_parsed.max()
                        if pd.isna(t_min) or pd.isna(t_max):
                            t_min, t_max = None, None
                        else:
                            t_min = (t_min - timedelta(hours=24)).to_pydatetime()
                            t_max = t_max.to_pydatetime()
                    except Exception:
                        t_min, t_max = None, None
                    all_headlines = _headlines_for_window(cur, t_min, t_max, limit=2000) if (t_min and t_max) else []

                    def _parse_ts(ts_val):
                        if ts_val is None:
                            return None
                        if hasattr(ts_val, "timestamp"):
                            return ts_val
                        try:
                            return pd.to_datetime(ts_val, utc=True).to_pydatetime()
                        except Exception:
                            return None

                    # Trade-impact bars (any position change, threshold 0.01; API can filter)
                    pnl = metrics_d.get("pnl_series", []) if metrics_d else []
                    delta = metrics_d.get("net_delta_series", []) if metrics_d else []
                    vega = metrics_d.get("net_vega_series", []) if metrics_d else []
                    delta = delta or [0] * len(bars)
                    vega = vega or [0] * len(bars)
                    pnl = pnl or [0] * len(bars)
                    threshold = 0.01
                    action_bars = []
                    for i in range(1, min(len(bars), len(delta), len(vega), len(pnl))):
                        d_prev, d_cur = delta[i - 1] or 0, delta[i] or 0
                        v_prev, v_cur = vega[i - 1] or 0, vega[i] or 0
                        if abs(d_cur - d_prev) >= threshold or abs(v_cur - v_prev) >= threshold:
                            bar = bars[i] if i < len(bars) else {}
                            bar_ts = bar.get("ts")
                            bar_dt = _parse_ts(bar_ts)
                            # Headlines in window [bar_ts - 24h, bar_ts] (most recent first, max 5)
                            headlines = []
                            if bar_dt and all_headlines:
                                window_end = bar_dt
                                window_start = bar_dt - timedelta(hours=24)
                                for h in all_headlines:
                                    ht = h.get("ts")
                                    try:
                                        h_dt = pd.to_datetime(ht, utc=True).to_pydatetime() if ht else None
                                    except Exception:
                                        h_dt = None
                                    if h_dt and window_start <= h_dt <= window_end:
                                        headlines.append(dict(h))
                                        if len(headlines) >= 5:
                                            break
                            action_bars.append({
                                "bar_ix": i,
                                "ts": _serialize_ts(bar_ts),
                                "pm_p": bar.get("pm_p"),
                                "pm_delta_p_1h": bar.get("pm_delta_p_1h"),
                                "sent_news_asset": bar.get("sent_news_asset"),
                                "sent_macro_topic": bar.get("sent_macro_topic"),
                                "atm_iv_30d": bar.get("atm_iv_30d"),
                                "delta_change": float(d_cur - d_prev),
                                "vega_change": float(v_cur - v_prev),
                                "pnl_this": float(pnl[i]) if i < len(pnl) else 0.0,
                                "headlines": headlines,
                            })
                    cur.execute(
                        """
                        INSERT INTO dashboard_trade_impact (underlying, updated_at, bars)
                        VALUES (%s, NOW(), %s::jsonb)
                        ON CONFLICT (underlying) DO UPDATE SET updated_at = NOW(), bars = EXCLUDED.bars
                        """,
                        (underlying, json.dumps(action_bars)),
                    )

    print("Dashboard snapshot written successfully.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Write dashboard snapshot to DB")
    p.add_argument("--task-run-id", type=int, default=None, help="Task run ID for progress tracking")
    _args = p.parse_args()
    main()
