#!/usr/bin/env python3
"""
Run the complete pipeline end-to-end for integration testing.
Steps: (1) migrations, (2) equity backfill, (3) options backfill, (4) PM backfill,
(5) sentiment backfill, (6) score sentiment, (7) build PM features, (8) options features,
(9) sentiment features, (10) feature matrix, (11) baselines, (12) ablation smoke test,
(13) generate reports.
Use --dry-run to print steps without running. Use --steps 1,2,10 to run only those.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# Step names and short descriptions for logging
STEPS = [
    (1, "apply_migrations", "Apply all migrations"),
    (2, "backfill_equity", "Backfill equity bars (--years 2)"),
    (3, "backfill_options", "Backfill options snapshots"),
    (4, "backfill_pm", "Backfill prediction market data"),
    (5, "backfill_sentiment", "Backfill sentiment (news, reddit if configured)"),
    (6, "score_sentiment", "Score sentiment docs"),
    (7, "build_pm_features", "Build PM features"),
    (8, "build_options_features", "Build options features"),
    (9, "build_sentiment_features", "Build sentiment features"),
    (10, "build_feature_matrix", "Build feature matrix"),
    (11, "run_baselines", "Run baselines evaluation"),
    (12, "run_ablation_smoke", "Run short ablation (1 seed, 1000 steps, PPO only)"),
    (13, "generate_reports", "Generate reports"),
]

# Tables to report row counts (order from migrations)
TABLE_NAMES = [
    "pm_events",
    "pm_markets",
    "pm_prices",
    "pm_features",
    "equity_bars",
    "options_snapshots",
    "options_features",
    "sentiment_docs",
    "sentiment_scored",
    "sentiment_features",
    "feature_bars",
]


def _log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def _get_table_counts() -> dict[str, int]:
    """Return row count per table (0 if table missing or error)."""
    try:
        from src.db import get_connection
        counts = {}
        with get_connection() as conn:
            with conn.cursor() as cur:
                for t in TABLE_NAMES:
                    try:
                        cur.execute(f"SELECT count(*) FROM {t}")
                        counts[t] = cur.fetchone()[0]
                    except Exception:
                        counts[t] = 0
        return counts
    except Exception:
        return {t: 0 for t in TABLE_NAMES}


def step_1_apply_migrations() -> dict:
    from src.db import apply_migrations
    apply_migrations()
    return {"migrations": "applied"}


def step_2_backfill_equity(years: int = 2) -> dict:
    from src.ingestion.backfill_equity import run_backfill_equity
    run_backfill_equity(years=years, symbols=None)
    counts = _get_table_counts()
    return {"equity_bars": counts.get("equity_bars", 0)}


def step_3_backfill_options() -> dict:
    from src.ingestion.backfill_options import run_backfill_options
    run_backfill_options(symbols=None, snapshot_date=None, use_tradier=False)
    counts = _get_table_counts()
    return {"options_snapshots": counts.get("options_snapshots", 0)}


def step_4_backfill_pm() -> dict:
    from src.ingestion.backfill_pm import run_backfill
    run_backfill(
        polymarket=True,
        kalshi=True,
        max_events_per_platform=25,
        days_history=365,
    )
    counts = _get_table_counts()
    return {
        "pm_events": counts.get("pm_events", 0),
        "pm_markets": counts.get("pm_markets", 0),
        "pm_prices": counts.get("pm_prices", 0),
    }


def step_5_backfill_sentiment(days: int = 7) -> dict:
    from src.connectors.sentiment import NewsAPICollector, RedditCollector
    from src.db import get_connection
    from src.ingestion.sentiment_writer import write_sentiment_docs
    since_ts = datetime.now(timezone.utc) - timedelta(days=days)
    documents = []
    try:
        documents.extend(NewsAPICollector().fetch(since_ts))
    except Exception as e:
        _log(f"NewsAPI skip: {e}")
    try:
        documents.extend(RedditCollector().fetch(since_ts))
    except Exception as e:
        _log(f"Reddit skip: {e}")
    written = 0
    if documents:
        with get_connection() as conn:
            written = write_sentiment_docs(conn, documents)
    return {"sentiment_docs_written": written}


def step_6_score_sentiment() -> dict:
    from src.connectors.sentiment import score_documents
    from src.db import get_connection
    from src.ingestion.sentiment_writer import read_sentiment_docs, write_sentiment_scored
    with get_connection() as conn:
        docs = read_sentiment_docs(conn, only_unscored=True, limit=None)
    if not docs:
        return {"scored": 0}
    scored = score_documents(docs, model="auto")
    with get_connection() as conn:
        written = write_sentiment_scored(conn, scored)
    return {"scored": written}


def step_7_build_pm_features() -> dict:
    from src.features.pm_feature_builder import run_build_pm_features
    run_build_pm_features(token_id=None, schema_version=1)
    counts = _get_table_counts()
    return {"pm_features": counts.get("pm_features", 0)}


def step_8_build_options_features() -> dict:
    from src.features.options_features import run_build_options_features
    run_build_options_features(underlying=None, feature_date=None, schema_version=1)
    counts = _get_table_counts()
    return {"options_features": counts.get("options_features", 0)}


def step_9_build_sentiment_features() -> dict:
    from src.features.sentiment_features import run_build_sentiment_features
    run_build_sentiment_features(underlying=None, start_date=None, end_date=None, schema_version=1)
    counts = _get_table_counts()
    return {"sentiment_features": counts.get("sentiment_features", 0)}


def step_10_build_feature_matrix() -> dict:
    from src.features.align import run_build_feature_matrix
    run_build_feature_matrix(underlying=None, start_date=None, end_date=None, schema_version=1)
    counts = _get_table_counts()
    return {"feature_bars": counts.get("feature_bars", 0)}


def step_11_run_baselines() -> dict:
    from scripts.run_baselines import main as run_baselines_main
    run_baselines_main(underlying="SPY", limit=2000, n_episodes=2, out_json="pipeline_baselines.json")
    return {"baselines": "done"}


def step_12_run_ablation_smoke() -> dict:
    from src.agents.ablation import run_ablation, save_ablation_results
    from src.envs.options_env import load_feature_bars_from_db
    feature_bars = load_feature_bars_from_db(underlying="SPY", limit=1500)
    if not feature_bars:
        raise RuntimeError("No feature_bars; run steps 1–10 first")
    out = run_ablation(
        feature_bars=feature_bars,
        algorithms=("ppo",),
        seeds=[0],
        total_timesteps=1000,
        models_dir=Path("models"),
        train_pct=0.70,
        val_pct=0.15,
    )
    save_ablation_results(out, json_path="ablation_results.json", csv_path="ablation_results.csv")
    return {"ablation_runs": len(out.get("results", []))}


def step_13_generate_reports() -> dict:
    from scripts.generate_reports import main as generate_reports_main
    generate_reports_main(
        ablation_json="ablation_results.json",
        output_dir="reports",
        feature_bars_db=None,
        models_dir="models",
        limit_bars=1000,
    )
    return {"reports": "done"}


def run_step(step_num: int, dry_run: bool, step_args: dict) -> tuple[bool, dict, str | None]:
    """Run one step. Returns (success, info_dict, error_message)."""
    name = next((n for i, n, _ in STEPS if i == step_num), None)
    if not name:
        return False, {}, f"Unknown step {step_num}"
    if dry_run:
        return True, {"dry_run": True}, None
    start = time.time()
    try:
        if step_num == 1:
            info = step_1_apply_migrations()
        elif step_num == 2:
            info = step_2_backfill_equity(years=step_args.get("equity_years", 2))
        elif step_num == 3:
            info = step_3_backfill_options()
        elif step_num == 4:
            info = step_4_backfill_pm()
        elif step_num == 5:
            info = step_5_backfill_sentiment(days=step_args.get("sentiment_days", 7))
        elif step_num == 6:
            info = step_6_score_sentiment()
        elif step_num == 7:
            info = step_7_build_pm_features()
        elif step_num == 8:
            info = step_8_build_options_features()
        elif step_num == 9:
            info = step_9_build_sentiment_features()
        elif step_num == 10:
            info = step_10_build_feature_matrix()
        elif step_num == 11:
            info = step_11_run_baselines()
        elif step_num == 12:
            info = step_12_run_ablation_smoke()
        elif step_num == 13:
            info = step_13_generate_reports()
        else:
            return False, {}, f"Unknown step {step_num}"
        elapsed = time.time() - start
        info["_elapsed_sec"] = round(elapsed, 2)
        return True, info, None
    except Exception as e:
        elapsed = time.time() - start
        return False, {"_elapsed_sec": round(elapsed, 2)}, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline end-to-end (integration test)")
    parser.add_argument("--dry-run", action="store_true", help="Print steps only, do not execute")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers to run (e.g. 1,2,3,10). Default: all.",
    )
    parser.add_argument("--equity-years", type=int, default=2, help="Years for equity backfill (step 2)")
    parser.add_argument("--sentiment-days", type=int, default=7, help="Days for sentiment backfill (step 5)")
    args = parser.parse_args()

    step_nums = list(range(1, 14))
    if args.steps:
        try:
            step_nums = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
        except ValueError:
            print("Invalid --steps; use e.g. --steps 1,2,3,10", file=sys.stderr)
            sys.exit(1)

    step_args = {"equity_years": args.equity_years, "sentiment_days": args.sentiment_days}
    results = []
    _log("Pipeline start (dry_run=%s, steps=%s)" % (args.dry_run, step_nums))

    for step_num in step_nums:
        step_def = next((s for s in STEPS if s[0] == step_num), None)
        if not step_def:
            _log("Step %d: unknown, skip" % step_num)
            results.append((step_num, False, {}, "Unknown step"))
            continue
        _, name, desc = step_def
        _log("Step %d %s: start - %s" % (step_num, name, desc))
        ok, info, err = run_step(step_num, args.dry_run, step_args)
        if err:
            _log("Step %d %s: FAILED - %s" % (step_num, name, err))
            results.append((step_num, False, info, err))
        else:
            elapsed = info.get("_elapsed_sec", 0)
            _log("Step %d %s: done in %s s - %s" % (step_num, name, elapsed, info))
            results.append((step_num, True, info, None))

    # Summary
    _log("")
    _log("========== Pipeline summary ==========")
    succeeded = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]
    _log("Succeeded: %s" % (succeeded or "none"))
    _log("Failed: %s" % (failed or "none"))
    for r in results:
        step_num, ok, info, err = r
        name = next((n for i, n, _ in STEPS if i == step_num), "?")
        if err:
            _log("  Step %d %s: %s" % (step_num, name, err))
        else:
            _log("  Step %d %s: %s" % (step_num, name, info))

    # Table row counts (always show when not dry-run)
    if not args.dry_run:
        try:
            counts = _get_table_counts()
            _log("")
            _log("========== DB table row counts ==========")
            for t in TABLE_NAMES:
                _log("  %s: %s" % (t, counts.get(t, 0)))
        except Exception as e:
            _log("Could not fetch table counts: %s" % e)

    # Append one-line summary to pipeline_runs.log
    if not args.dry_run:
        try:
            log_path = ROOT / "pipeline_runs.log"
            ts = datetime.now().isoformat(timespec="seconds")
            ok_steps = ",".join(str(s) for s in succeeded) or "none"
            fail_steps = ",".join(str(s) for s in failed) or "none"
            status = "OK" if not failed else "PARTIAL"
            line = f"{ts} | {status} | steps_ok=[{ok_steps}] steps_fail=[{fail_steps}]\n"
            with open(log_path, "a") as f:
                f.write(line)
        except Exception as e:
            _log(f"Could not write pipeline_runs.log: {e}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
