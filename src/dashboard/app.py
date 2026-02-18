"""
Streamlit dashboard: Daily Monitor, Performance, Exposure, Signal Inspector, Trade Log, Ablation, Event Case Studies.
Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root on path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

VARIANT_LABELS = {"A": "Base", "B": "+Sentiment", "C": "+PM", "D": "Full"}
BASELINE_NAMES = ["BuyAndHold", "FixedLongVol", "SimpleEventRule", "DeltaNeutral", "RandomPolicy"]

# Tables tracked by the pipeline
_MONITOR_TABLES = [
    ("pm_events", "ts", "start_ts"),
    ("pm_markets", "ts", "created_at"),
    ("pm_prices", "ts", "ts"),
    ("pm_features", "ts", "ts"),
    ("equity_bars", "date", "ts"),
    ("options_snapshots", "date", "snapshot_date"),
    ("options_features", "date", "feature_date"),
    ("sentiment_docs", "ts", "ts"),
    ("sentiment_scored", "ts", "ts"),
    ("sentiment_features", "ts", "ts"),
    ("feature_bars", "ts", "ts"),
]


def _safe_query(cur, sql: str, params=None) -> list:
    try:
        cur.execute(sql, params or ())
        return cur.fetchall()
    except Exception:
        return []


# ---------- Page 0: Daily Monitor ----------


def page_daily_monitor():
    st.subheader("Daily Monitor")
    if st.button("Refresh", key="refresh_monitor"):
        st.rerun()

    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                _section_data_accumulation(cur)
                st.divider()
                _section_feature_coverage_heatmap(cur)
                st.divider()
                col_left, col_right = st.columns(2)
                with col_left:
                    _section_pm_price_coverage(cur)
                with col_right:
                    _section_sentiment_volume(cur)
                st.divider()
                _section_options_snapshot_tracker(cur)
    except Exception as e:
        st.error(f"Cannot connect to DB: {e}")

    st.divider()
    _section_pipeline_run_log()


def _section_data_accumulation(cur):
    """Section 1: Data Accumulation Table."""
    st.markdown("#### Data Accumulation")
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_48h = now - timedelta(hours=48)

    rows_out = []
    for table_name, ts_kind, ts_col in _MONITOR_TABLES:
        # Total count
        total_rows = _safe_query(cur, f"SELECT count(*) FROM {table_name}")
        total = total_rows[0][0] if total_rows else 0

        # Recent rows and latest timestamp
        recent = 0
        latest_ts = None
        if ts_col:
            if ts_kind == "ts":
                res = _safe_query(cur, f"SELECT count(*) FROM {table_name} WHERE {ts_col} >= %s", (cutoff_24h,))
                recent = res[0][0] if res else 0
                res2 = _safe_query(cur, f"SELECT max({ts_col}) FROM {table_name}")
                latest_ts = res2[0][0] if res2 and res2[0][0] else None
            elif ts_kind == "date":
                cutoff_date = cutoff_24h.date()
                res = _safe_query(cur, f"SELECT count(*) FROM {table_name} WHERE {ts_col} >= %s", (cutoff_date,))
                recent = res[0][0] if res else 0
                res2 = _safe_query(cur, f"SELECT max({ts_col}) FROM {table_name}")
                latest_ts = res2[0][0] if res2 and res2[0][0] else None

        # Format latest timestamp
        if latest_ts is not None:
            if hasattr(latest_ts, "isoformat"):
                latest_str = str(latest_ts)[:19]
            else:
                latest_str = str(latest_ts)
        else:
            latest_str = "—"

        # Determine freshness status
        if total == 0:
            status = "empty"
        elif recent > 0:
            status = "fresh"
        elif latest_ts is not None:
            if hasattr(latest_ts, "date"):
                age = (now.date() - (latest_ts.date() if hasattr(latest_ts, "date") else latest_ts)).days
            else:
                age = 3
            status = "stale" if age <= 2 else "old"
        else:
            status = "old"

        rows_out.append({
            "Table": table_name,
            "Total Rows": f"{total:,}",
            "Added (24h)": f"+{recent:,}" if recent else "0",
            "Latest": latest_str,
            "Status": status,
        })

    df = pd.DataFrame(rows_out)

    def _color_status(val):
        colors = {"fresh": "background-color: #2d6a2d; color: white",
                  "stale": "background-color: #8a6d00; color: white",
                  "old": "background-color: #8a2020; color: white",
                  "empty": "background-color: #555; color: #ccc"}
        return colors.get(val, "")

    styled = df.style.map(_color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=440)


def _section_feature_coverage_heatmap(cur):
    """Section 2: Feature Coverage Heatmap for last 7 days."""
    st.markdown("#### Feature Coverage Heatmap (last 7 days)")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date()
    rows = _safe_query(cur, """
        SELECT underlying, ts::date AS bar_date,
               atm_iv_7d, atm_iv_30d, iv_skew, iv_term_slope,
               sent_news_asset, sent_social_asset, sent_macro_topic,
               pm_p, pm_logit_p, pm_delta_p_1h,
               equity_return_1d, equity_realized_vol_20d
        FROM feature_bars
        WHERE ts::date >= %s
        ORDER BY underlying, ts::date
    """, (cutoff,))

    if not rows:
        st.info("No feature_bars in the last 7 days.")
        return

    feature_names = [
        "atm_iv_7d", "atm_iv_30d", "iv_skew", "iv_term_slope",
        "sent_news", "sent_social", "sent_macro",
        "pm_p", "pm_logit_p", "pm_delta_1h",
        "eq_return", "eq_vol_20d",
    ]
    # Aggregate per (underlying, date): for each feature, compute fraction non-null and non-zero
    from collections import defaultdict
    agg: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r[0], str(r[1]))
        vals = list(r[2:])
        for i, fn in enumerate(feature_names):
            v = vals[i]
            if v is None:
                agg[key][fn].append(-1)  # null
            elif float(v) == 0.0:
                agg[key][fn].append(0)   # zero
            else:
                agg[key][fn].append(1)   # populated

    # Build matrix: rows = (underlying, date), cols = features
    y_labels = []
    z_matrix = []
    for key in sorted(agg.keys()):
        y_labels.append(f"{key[0]} {key[1]}")
        row_vals = []
        for fn in feature_names:
            vals = agg[key][fn]
            if not vals:
                row_vals.append(-1)
            else:
                avg = sum(vals) / len(vals)
                if avg > 0.5:
                    row_vals.append(1)
                elif avg > -0.5:
                    row_vals.append(0)
                else:
                    row_vals.append(-1)
        z_matrix.append(row_vals)

    colorscale = [[0.0, "#c62828"], [0.5, "#9e9e9e"], [1.0, "#2e7d32"]]
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=feature_names,
        y=y_labels,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        showscale=False,
        hovertemplate="Feature: %{x}<br>Row: %{y}<br>Status: %{z}<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, len(y_labels) * 28 + 80),
        margin=dict(l=10, r=10, t=30, b=30),
        xaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = populated, Gray = zero, Red = null")


def _section_pm_price_coverage(cur):
    """Section 3: PM Price Coverage Chart (last 30 days)."""
    st.markdown("#### PM Prices (last 30 days)")
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    rows = _safe_query(cur, """
        SELECT ts::date AS day, platform, count(*) AS cnt
        FROM pm_prices
        WHERE ts >= %s
        GROUP BY day, platform
        ORDER BY day
    """, (cutoff,))

    if not rows:
        st.info("No pm_prices in the last 30 days.")
        return

    df = pd.DataFrame(rows, columns=["date", "platform", "count"])
    fig = px.line(df, x="date", y="count", color="platform",
                  markers=True, title="Daily PM Price Rows by Platform")
    fig.update_layout(height=320, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _section_sentiment_volume(cur):
    """Section 4: Sentiment Volume Chart (last 14 days)."""
    st.markdown("#### Sentiment Docs (last 14 days)")
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    rows = _safe_query(cur, """
        SELECT ts::date AS day, source, count(*) AS cnt
        FROM sentiment_docs
        WHERE ts >= %s
        GROUP BY day, source
        ORDER BY day
    """, (cutoff,))

    if not rows:
        st.info("No sentiment_docs in the last 14 days.")
        return

    df = pd.DataFrame(rows, columns=["date", "source", "count"])
    fig = px.bar(df, x="date", y="count", color="source",
                 barmode="group", title="Daily Sentiment Docs by Source")
    fig.update_layout(height=320, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _section_options_snapshot_tracker(cur):
    """Section 5: Options Snapshot Tracker."""
    st.markdown("#### Options Snapshot Tracker")
    rows = _safe_query(cur, """
        SELECT underlying,
               count(DISTINCT snapshot_date) AS unique_dates,
               min(snapshot_date) AS earliest,
               max(snapshot_date) AS latest,
               count(*) AS total_contracts
        FROM options_snapshots
        GROUP BY underlying
        ORDER BY underlying
    """)

    if not rows:
        st.info("No options_snapshots in DB yet. Run step 3 to backfill.")
        return

    df = pd.DataFrame(rows, columns=["Underlying", "Snapshot Days", "Earliest", "Latest", "Total Contracts"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("This should grow by 1 snapshot day each time the pipeline runs.")


def _section_pipeline_run_log():
    """Section 6: Pipeline Run Log."""
    st.markdown("#### Pipeline Run Log")
    log_path = ROOT / "pipeline_runs.log"
    if not log_path.exists():
        st.info("No pipeline_runs.log found. Run the pipeline to generate it.")
        return
    try:
        lines = log_path.read_text().strip().splitlines()
        last_lines = lines[-10:] if len(lines) >= 10 else lines
        st.code("\n".join(last_lines), language="text")
    except Exception as e:
        st.warning(f"Could not read pipeline_runs.log: {e}")


def _load_feature_bars(underlying: str, start_date: str | None, end_date: str | None, limit: int) -> list[dict]:
    try:
        from src.envs.options_env import load_feature_bars_from_db
        return load_feature_bars_from_db(underlying=underlying, start_date=start_date, end_date=end_date, limit=limit)
    except Exception as e:
        st.warning(f"Could not load feature_bars from DB: {e}")
        return []


def _load_ablation(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    try:
        from src.agents.analysis import load_ablation_results
        return load_ablation_results(path)
    except Exception as e:
        st.error(f"Could not load ablation JSON: {e}")
        return None


def _get_series_from_cache_or_file(session_key: str, variant_or_baseline: str, file_dir: str | None):
    """Get equity/pnl series from session_state (after Run evaluation) or from JSON file."""
    if session_key not in st.session_state:
        st.session_state[session_key] = {}
    cache = st.session_state[session_key]
    if variant_or_baseline in cache:
        return cache[variant_or_baseline]
    if file_dir:
        path = Path(file_dir) / f"series_{variant_or_baseline}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            cache[variant_or_baseline] = data
            return data
    return None


def _run_evaluation_and_cache(bars: list[dict], variant_or_baseline: str, underlying: str, seed: int) -> dict | None:
    """Run eval for one variant/baseline and return series dict; cache in session_state."""
    if not bars or len(bars) < 2:
        return None
    try:
        from src.envs.options_env import OptionsEnv
        from src.agents.eval import evaluate_policy_with_series
        from src.agents.obs_mask_wrapper import ObsMaskWrapper
        from src.agents.baselines import BuyAndHold, FixedLongVol, SimpleEventRule, DeltaNeutral, RandomPolicy
    except ImportError as e:
        st.error(f"Import error: {e}")
        return None

    env = OptionsEnv(feature_bars=bars, underlying=underlying)
    if variant_or_baseline in VARIANT_LABELS:
        try:
            from stable_baselines3 import PPO
            from src.agents.ablation import SB3PolicyAdapter
            v = variant_or_baseline
            model_path = Path("models") / f"ablation_{v}_ppo_seed{seed}.zip"
            if not model_path.exists():
                st.warning(f"Model not found: {model_path}")
                return None
            model = PPO.load(str(model_path))
            env = ObsMaskWrapper(env, variant=v)
            policy = SB3PolicyAdapter(model, algorithm="ppo")
        except Exception as e:
            st.error(f"Load model: {e}")
            return None
    else:
        policies = {
            "BuyAndHold": BuyAndHold(),
            "FixedLongVol": FixedLongVol(),
            "SimpleEventRule": SimpleEventRule(),
            "DeltaNeutral": DeltaNeutral(),
            "RandomPolicy": RandomPolicy(seed=seed),
        }
        policy = policies.get(variant_or_baseline)
        if policy is None:
            return None
    with st.spinner(f"Running evaluation: {variant_or_baseline}..."):
        metrics = evaluate_policy_with_series(env, policy, n_episodes=1, seeds=[seed])
    out = {
        "equity": metrics.get("equity_series", []),
        "pnl": metrics.get("pnl_series", []),
        "vix": metrics.get("vix_series", []),
        "net_delta": metrics.get("net_delta_series", []),
        "net_vega": metrics.get("net_vega_series", []),
    }
    if "eval_cache" not in st.session_state:
        st.session_state["eval_cache"] = {}
    st.session_state["eval_cache"][variant_or_baseline] = out
    return out


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(np.asarray(equity, dtype=float))
    return (peak - np.asarray(equity, dtype=float)) / np.maximum(peak, 1e-8)


def _rolling_sharpe(pnl: np.ndarray, equity: np.ndarray, window: int) -> np.ndarray:
    n = len(pnl)
    out = np.full(n, np.nan)
    pnl = np.asarray(pnl, dtype=float)
    equity = np.asarray(equity, dtype=float)
    for t in range(window - 1, n):
        eq = equity[t - window + 1 : t + 1]
        p = pnl[t - window + 1 : t + 1]
        ret = np.zeros(window)
        for i in range(window):
            denom = (eq[0] - p[0]) if i == 0 else eq[i - 1]
            ret[i] = p[i] / max(abs(denom), 1e-8)
        if np.std(ret) < 1e-12:
            out[t] = 0.0
        else:
            out[t] = float(np.mean(ret) / np.std(ret) * np.sqrt(252 * 27))
    return out


# ---------- Page 1: Performance Overview ----------
def page_performance_overview(
    bars: list[dict],
    underlying: str,
    ablation_path: str | None,
    series_dir: str | None,
    default_start: str,
    default_end: str,
):
    st.subheader("Performance Overview")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=pd.to_datetime(default_start).date() if default_start else None)
        end = st.date_input("End date", value=pd.to_datetime(default_end).date() if default_end else None)
    with col2:
        options = ["Variant A", "Variant B", "Variant C", "Variant D"] + BASELINE_NAMES
        selected = st.multiselect("Variant / Baseline", options, default=["Variant D", "BuyAndHold"])
    run_eval = st.button("Run evaluation for selected")
    seed = st.number_input("Seed", value=0, min_value=0, key="perf_seed")

    if not bars:
        st.info("Load feature bars (set date range and underlying in sidebar) to see performance.")
        return

    # Filter bars by date if we have ts
    bars_df = pd.DataFrame(bars) if bars else pd.DataFrame()
    try:
        if not bars_df.empty and "ts" in bars_df.columns:
            bars_df["ts"] = pd.to_datetime(bars_df["ts"])
            if start:
                bars_df = bars_df[bars_df["ts"].dt.date >= start]
            if end:
                bars_df = bars_df[bars_df["ts"].dt.date <= end]
            bars = bars_df.to_dict("records")
    except Exception:
        pass

    series_data = {}
    for name in selected:
        if name.startswith("Variant "):
            vob = name.replace("Variant ", "").strip()
            key = name
        else:
            vob = name
            key = name
        data = _get_series_from_cache_or_file("eval_cache", vob, series_dir)
        if run_eval and vob in (list(VARIANT_LABELS) + BASELINE_NAMES):
            data = _run_evaluation_and_cache(bars, vob, underlying, seed)
        if data and data.get("equity"):
            series_data[key] = dict(data)
            if not bars_df.empty and "ts" in bars_df.columns and len(bars_df) >= len(data["equity"]):
                ts = bars_df["ts"].iloc[1 : len(data["equity"]) + 1].tolist()
                if len(ts) == len(data["equity"]):
                    series_data[key]["ts"] = ts

    if not series_data:
        st.info("Run evaluation or provide series JSON directory to plot.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("Install plotly: pip install plotly")
        return

    # Equity curve
    fig = go.Figure()
    for label, d in series_data.items():
        eq = d["equity"]
        x = d.get("ts") or list(range(len(eq)))
        fig.add_trace(go.Scatter(x=x, y=eq, mode="lines", name=label))
    fig.update_layout(title="Equity curve", xaxis_title="Time", yaxis_title="Equity", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    fig2 = go.Figure()
    for label, d in series_data.items():
        eq = d["equity"]
        dd = _drawdown_series(np.array(eq))
        x = d.get("ts") or list(range(len(dd)))
        fig2.add_trace(go.Scatter(x=x, y=dd, mode="lines", name=label))
    fig2.update_layout(title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Rolling Sharpe (30-bar)
    fig3 = go.Figure()
    window = 30
    for label, d in series_data.items():
        pnl = d.get("pnl", [])
        eq = d.get("equity", [])
        if len(pnl) >= window and len(eq) >= window:
            rs = _rolling_sharpe(np.array(pnl), np.array(eq), window)
            x = d.get("ts") or list(range(len(rs)))
            fig3.add_trace(go.Scatter(x=x, y=rs, mode="lines", name=label))
    fig3.update_layout(title="Rolling Sharpe (30-bar)", xaxis_title="Time", yaxis_title="Sharpe", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # Cumulative P&L
    fig4 = go.Figure()
    for label, d in series_data.items():
        pnl = d.get("pnl", [])
        if pnl:
            cum = np.cumsum(pnl)
            x = d.get("ts") or list(range(len(cum)))
            fig4.add_trace(go.Scatter(x=x, y=cum, mode="lines", name=label))
    fig4.update_layout(title="Cumulative P&L", xaxis_title="Time", yaxis_title="Cumulative P&L", height=350)
    st.plotly_chart(fig4, use_container_width=True)


# ---------- Page 2: Exposure Monitor ----------
def page_exposure_monitor(bars: list[dict], series_dir: str | None, default_start: str, default_end: str):
    st.subheader("Exposure Monitor")
    start = st.date_input("Start date", value=pd.to_datetime(default_start).date() if default_start else None, key="exp_start")
    end = st.date_input("End date", value=pd.to_datetime(default_end).date() if default_end else None, key="exp_end")
    # Use Variant D series if available
    data = _get_series_from_cache_or_file("eval_cache", "D", series_dir)
    if not data:
        st.info("Run evaluation for Variant D in Performance Overview, or provide series JSON for Variant D.")
        return
    delta = data.get("net_delta", [])
    vega = data.get("net_vega", [])
    if not delta and not vega:
        st.info("No delta/vega series in cache.")
        return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    n = max(len(delta), len(vega))
    x = list(range(n))
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Net delta", "Net vega"), shared_x=True)
    fig.add_trace(go.Scatter(x=x, y=delta or [0] * n, name="Net delta"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=vega or [0] * n, name="Net vega"), row=2, col=1)
    fig.update_layout(height=500, title="Exposure over time")
    st.plotly_chart(fig, use_container_width=True)
    # Gamma: env doesn't expose per-step gamma; show placeholder or 0
    st.caption("Gamma over time: not recorded in current eval output (placeholder).")
    pos_count = [1 if (d or 0) != 0 or (v or 0) != 0 else 0 for d, v in zip(delta, vega)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=pos_count, mode="lines", name="Position count (proxy)"))
    fig2.update_layout(title="Position count (proxy)", xaxis_title="Bar", height=300)
    st.plotly_chart(fig2, use_container_width=True)


# ---------- Page 3: Signal Inspector ----------
def page_signal_inspector(bars: list[dict]):
    st.subheader("Signal Inspector")
    if not bars:
        st.info("Load feature bars to inspect signals.")
        return
    df = pd.DataFrame(bars)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        options_ts = df["ts"].astype(str).tolist()
    else:
        options_ts = [str(i) for i in range(len(df))]
    bar_choice = st.selectbox("Select bar (time or index)", options_ts, index=min(len(options_ts) // 2, len(options_ts) - 1))
    try:
        idx = options_ts.index(bar_choice)
    except ValueError:
        idx = 0
    row = bars[idx]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Polymarket / prediction markets**")
        st.metric("P (event)", row.get("pm_p") or "—")
        st.metric("Δp 1h", row.get("pm_delta_p_1h") or "—")
        st.metric("Δp 1d", row.get("pm_delta_p_1d") or "—")
        st.metric("Vol of p", row.get("pm_vol_of_p") or "—")
    with col2:
        st.markdown("**Sentiment**")
        st.metric("News (asset)", row.get("sent_news_asset") or "—")
        st.metric("Social (asset)", row.get("sent_social_asset") or "—")
        st.metric("Macro topic", row.get("sent_macro_topic") or "—")
        st.metric("Dispersion", row.get("sent_dispersion") or "—")
        st.metric("Momentum", row.get("sent_momentum") or "—")
    st.markdown("**IV surface snapshot**")
    iv_df = pd.DataFrame({
        "Expiry": ["7d", "14d", "30d"],
        "ATM IV": [row.get("atm_iv_7d"), row.get("atm_iv_14d"), row.get("atm_iv_30d")],
    })
    st.dataframe(iv_df, use_container_width=True, hide_index=True)
    st.metric("IV term slope", row.get("iv_term_slope") or "—")
    st.metric("IV skew", row.get("iv_skew") or "—")


# ---------- Page 4: Trade Log ----------
def page_trade_log(trades_path: str | None):
    st.subheader("Trade Log")
    if trades_path and Path(trades_path).exists():
        try:
            with open(trades_path) as f:
                raw = json.load(f)
            trades = raw if isinstance(raw, list) else raw.get("trades", [])
            df = pd.DataFrame(trades)
        except Exception as e:
            st.warning(f"Could not load trades: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "contract", "direction", "entry_price", "exit_price",
            "pnl", "delta_entry", "vega_entry", "holding_period_bars",
        ])
    if df.empty:
        st.info("No trade data. Provide a JSON file with a 'trades' list (entry_time, exit_time, contract, direction, entry_price, exit_price, pnl, greeks, etc.).")
        return
    st.dataframe(df, use_container_width=True, height=400)
    st.download_button("Download as CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="trades.csv", mime="text/csv")


# ---------- Page 5: Ablation Comparison ----------
def page_ablation_comparison(ablation_data: dict | None):
    st.subheader("Ablation Comparison")
    if not ablation_data:
        st.info("Load ablation results JSON in the sidebar.")
        return
    agg = [a for a in ablation_data.get("aggregated", []) if a.get("algorithm") == "ppo"]
    pvals = ablation_data.get("pvalues_vs_A", {})
    if not agg:
        st.info("No aggregated results in ablation JSON.")
        return
    import plotly.graph_objects as go
    variants = [VARIANT_LABELS.get(a["variant"], a["variant"]) for a in agg]
    sharpe_mean = [a["sharpe_mean"] for a in agg]
    sharpe_std = [a["sharpe_std"] for a in agg]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=variants, y=sharpe_mean, error_y=dict(type="data", array=sharpe_std), name="Sharpe"))
    fig.update_layout(title="Sharpe ratio (mean ± std)", xaxis_title="Variant", height=400)
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=variants, y=[a["max_drawdown_mean"] for a in agg], error_y=dict(type="data", array=[a["max_drawdown_std"] for a in agg]), name="Max DD"))
    fig2.update_layout(title="Max drawdown", xaxis_title="Variant", height=350)
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=variants, y=[a["hit_rate_mean"] for a in agg], error_y=dict(type="data", array=[a["hit_rate_std"] for a in agg]), name="Hit rate %"))
    fig3.update_layout(title="Hit rate (%)", xaxis_title="Variant", height=350)
    st.plotly_chart(fig3, use_container_width=True)
    # Table with p-values
    rows = []
    for a in agg:
        v = a["variant"]
        pkey = f"pval_sharpe_vs_A_{v}_ppo"
        pval = pvals.get(pkey, float("nan"))
        pval_str = f"{pval:.4f}" if not (isinstance(pval, float) and np.isnan(pval)) else "—"
        if v == "A":
            pval_str = "—"
        rows.append({
            "Variant": VARIANT_LABELS.get(v, v),
            "Sharpe": f"{a['sharpe_mean']:.4f} ± {a['sharpe_std']:.4f}",
            "Max DD": f"{a['max_drawdown_mean']:.4f} ± {a['max_drawdown_std']:.4f}",
            "Hit rate (%)": f"{a['hit_rate_mean']:.2f} ± {a['hit_rate_std']:.2f}",
            "p vs A": pval_str,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------- Page 6: Event Case Studies ----------
def page_event_case_studies(bars: list[dict], models_dir: str, events: list[str]):
    st.subheader("Event Case Studies")
    if not bars or len(bars) < 25:
        st.info("Load feature bars (at least 25 bars) to generate case studies.")
        return
    event = st.selectbox("Select event", events, index=0)
    # Map event to start_bar (e.g. event_1 -> 0, event_2 -> n//3, event_3 -> 2*n//3)
    event_to_start = {e: max(0, (len(bars) * (i // 3)) // 3 - 10) for i, e in enumerate(events)}
    start_bar = event_to_start.get(event, 0)
    if st.button("Generate case study figure"):
        try:
            from src.agents.analysis import generate_event_case_study
            model_path = Path(models_dir) / "ablation_D_ppo_seed0.zip"
            if not model_path.exists():
                st.warning(f"Model not found: {model_path}. Using baseline policy for demo.")
                from src.agents.baselines import DeltaNeutral
                model = DeltaNeutral()
            else:
                from stable_baselines3 import PPO
                model = PPO.load(str(model_path))
            fig = generate_event_case_study(bars, model, event, window_bars=20, start_bar=start_bar, output_path=None)
            if fig is not None:
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
            else:
                st.info("Case study figure could not be generated (e.g. matplotlib not available).")
        except Exception as e:
            st.error(str(e))


def main():
    st.set_page_config(page_title="Options Agent Dashboard", layout="wide")
    st.sidebar.title("Options Agent Dashboard")
    page = st.sidebar.radio(
        "Page",
        [
            "Daily Monitor",
            "Performance Overview",
            "Exposure Monitor",
            "Signal Inspector",
            "Trade Log",
            "Ablation Comparison",
            "Event Case Studies",
        ],
    )

    if page == "Daily Monitor":
        page_daily_monitor()
        return

    # Data options (only shown for non-monitor pages)
    st.sidebar.subheader("Data")
    underlying = st.sidebar.text_input("Underlying", value="SPY")
    limit_bars = st.sidebar.number_input("Max bars", value=2000, min_value=100)
    use_db = st.sidebar.checkbox("Load feature_bars from DB", value=True)
    start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="")
    end_date = st.sidebar.text_input("End date (YYYY-MM-DD)", value="")
    ablation_path = st.sidebar.text_input("Ablation JSON path", value="ablation_results.json")
    series_dir = st.sidebar.text_input("Series JSON directory (optional)", value="")
    trades_path = st.sidebar.text_input("Trades JSON path (optional)", value="")
    models_dir = st.sidebar.text_input("Models directory", value="models")

    # Load feature bars
    bars = []
    if use_db:
        bars = _load_feature_bars(underlying, start_date or None, end_date or None, limit_bars)
    default_start = start_date or (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    default_end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    ablation_data = _load_ablation(ablation_path)

    if page == "Performance Overview":
        page_performance_overview(bars, underlying, ablation_path, series_dir or None, default_start, default_end)
    elif page == "Exposure Monitor":
        page_exposure_monitor(bars, series_dir or None, default_start, default_end)
    elif page == "Signal Inspector":
        page_signal_inspector(bars)
    elif page == "Trade Log":
        page_trade_log(trades_path or None)
    elif page == "Ablation Comparison":
        page_ablation_comparison(ablation_data)
    elif page == "Event Case Studies":
        page_event_case_studies(bars, models_dir, ["event_1", "event_2", "event_3"])


if __name__ == "__main__":
    main()
