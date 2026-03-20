"""
Streamlit dashboard: Overview, Market Signals, Performance, Trade Impact.
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

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

VARIANT_LABELS = {"A": "Base", "B": "+Sentiment", "C": "+PM", "D": "Full"}
BASELINE_NAMES = ["BuyAndHold", "FixedLongVol", "SimpleEventRule", "DeltaNeutral", "RandomPolicy"]

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

# ---- Custom CSS ----
CUSTOM_CSS = """
<style>
.dashboard-card {
    background: var(--secondaryBackgroundColor, #262730);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.dashboard-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
    color: var(--textColor, #fafafa);
    font-weight: 500;
    opacity: 0.9;
}
.dashboard-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--textColor, #fafafa);
}
.dashboard-card .value.positive { color: #2e7d32; }
.dashboard-card .value.negative { color: #c62828; }
.dashboard-card .value.neutral { color: #1565c0; }
.section-spacer { margin-top: 1.5rem; }
.pipeline-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
.pipeline-dot.fresh { background: #2e7d32; }
.pipeline-dot.stale { background: #8a6d00; }
.pipeline-dot.old { background: #c62828; }
.pipeline-dot.empty { background: #555; }
.sentiment-bar { height: 8px; border-radius: 4px; margin: 4px 0; }
.sentiment-bar.positive { background: #2e7d32; }
.sentiment-bar.negative { background: #c62828; }
.sentiment-bar.neutral { background: #1565c0; }
</style>
"""


def _inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{float(x) * 100:.1f}%"


def _fmt_num(x: float | None, decimals: int = 2) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{decimals}f}"


def _fmt_ts(ts: Any) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "strftime"):
        return ts.strftime("%b %d, %I:%M %p")
    s = str(ts)
    if "T" in s:
        try:
            dt = pd.to_datetime(s)
            return dt.strftime("%b %d, %I:%M %p")
        except Exception:
            return s[:19]
    return s


def _safe_query(cur, sql: str, params=None) -> list:
    try:
        cur.execute(sql, params or ())
        return cur.fetchall()
    except Exception:
        return []




# ---------- Page 1: Overview ----------
def page_overview():
    _inject_css()
    st.subheader("Overview")
    if st.button("Refresh", key="overview_refresh"):
        st.rerun()

    try:
        from src.db import get_connection
        with st.spinner("Loading dashboard data…"):
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # KPI row
                    total_bars = _safe_query(cur, "SELECT count(*) FROM feature_bars")
                    total_prices = _safe_query(cur, "SELECT count(*) FROM pm_prices")
                    total_sentiment = _safe_query(cur, "SELECT count(*) FROM sentiment_scored")
                    total_events = _safe_query(cur, "SELECT count(*) FROM pm_events")
                    n_bars = total_bars[0][0] if total_bars else 0
                    n_prices = total_prices[0][0] if total_prices else 0
                    n_sentiment = total_sentiment[0][0] if total_sentiment else 0
                    n_events = total_events[0][0] if total_events else 0

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Feature bars", f"{n_bars:,}")
                    with col2:
                        st.metric("PM price points", f"{n_prices:,}")
                    with col3:
                        st.metric("PM events", f"{n_events:,}")
                    with col4:
                        st.metric("Scored headlines", f"{n_sentiment:,}")
                    with col5:
                        # Data freshness: latest feature_bars ts
                        res = _safe_query(cur, "SELECT max(ts) FROM feature_bars")
                        latest = res[0][0] if res and res[0][0] else None
                        st.metric("Latest data", _fmt_ts(latest) if latest else "—")

                    now = datetime.now(timezone.utc)
                    cutoff_24h = now - timedelta(hours=24)
                    statuses = []
                    for table_name, ts_kind, ts_col in _MONITOR_TABLES:
                        total_rows = _safe_query(cur, f"SELECT count(*) FROM {table_name}")
                        total = total_rows[0][0] if total_rows else 0
                        recent = 0
                        latest_ts = None
                        if ts_col:
                            if ts_kind == "ts":
                                res = _safe_query(cur, f"SELECT count(*), max({ts_col}) FROM {table_name} WHERE {ts_col} >= %s", (cutoff_24h,))
                                if res:
                                    recent, latest_ts = res[0][0], res[0][1]
                            else:
                                res = _safe_query(cur, f"SELECT count(*), max({ts_col}) FROM {table_name} WHERE {ts_col} >= %s", (cutoff_24h.date(),))
                                if res:
                                    recent, latest_ts = res[0][0], res[0][1]
                        if total == 0:
                            status = "empty"
                        elif recent > 0:
                            status = "fresh"
                        elif latest_ts and hasattr(latest_ts, "date"):
                            age = (now.date() - latest_ts.date()).days
                            status = "stale" if age <= 2 else "old"
                        else:
                            status = "old"
                        statuses.append((table_name, status))
                    line = ""
                    for name, status in statuses:
                        line += f'<span class="pipeline-dot {status}" title="{name}"></span>'
                    st.markdown(
                        '<div class="dashboard-card section-spacer"><h4>Pipeline status</h4><p>' + line + " <code>pm_events</code> · <code>pm_markets</code> · …</p><p style='font-size:0.85rem;opacity:0.85'>Green = fresh (24h), Yellow = stale (1–2d), Red = old, Gray = empty</p></div>",
                        unsafe_allow_html=True,
                    )

                    cutoff = (now - timedelta(days=7)).date()
                    rows = _safe_query(cur, """
                        SELECT count(*), count(atm_iv_7d), count(pm_p), count(sent_news_asset), count(equity_return_1d)
                        FROM feature_bars WHERE ts::date >= %s
                    """, (cutoff,))
                    if rows and rows[0][0] > 0:
                        total, iv, pm, sent, eq = rows[0]
                        cov_text = f"Bars: <strong>{total}</strong> · IV: <strong>{iv}</strong> · PM: <strong>{pm}</strong> · Sentiment: <strong>{sent}</strong> · Equity: <strong>{eq}</strong>"
                        st.markdown(f'<div class="dashboard-card"><h4>Feature coverage (last 7 days)</h4><p>{cov_text}</p></div>', unsafe_allow_html=True)
                    else:
                        st.info("No feature_bars in the last 7 days.")

                    log_path = ROOT / "pipeline_runs.log"
                    last_display = ""
                    if log_path.exists():
                        lines = log_path.read_text().strip().splitlines()
                        last = lines[-1] if lines else ""
                        last_display = (last[:80] + "…" if len(last) > 80 else last).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    model_rows = _safe_query(cur, "SELECT sentiment_model, count(*) FROM sentiment_scored GROUP BY sentiment_model")
                    model_summary = ", ".join(f"{m or 'unknown'}: {c}" for m, c in model_rows) if model_rows else "—"
                    st.markdown(
                        '<div class="dashboard-card"><h4>Last pipeline run</h4><pre style="margin:0;font-size:0.85rem;overflow-x:auto">' + (last_display or "—") + '</pre><p style="margin-top:0.5rem;font-size:0.85rem;opacity:0.9">Sentiment: ' + model_summary + "</p></div>",
                        unsafe_allow_html=True,
                    )
    except Exception as e:
        st.error("Database unavailable. Check DATABASE_URL and that Postgres is running.")
        st.exception(e)


# ---------- Page 2: Market Signals ----------
def page_market_signals():
    _inject_css()
    st.subheader("Market Signals")
    underlying = st.sidebar.text_input("Underlying", value="SPY", key="sig_underlying")
    limit = st.sidebar.number_input("Max bars", value=500, min_value=50, key="sig_bars")

    try:
        with st.spinner("Loading feature bars…"):
            from src.envs.options_env import load_feature_bars_from_db
            bars = load_feature_bars_from_db(underlying=underlying, limit=limit)
    except Exception as e:
        st.warning("Could not load feature bars. Run pipeline steps 1–10 and check underlying symbol.")
        st.caption(str(e))
        bars = []

    try:
        from src.db import get_connection
        with st.spinner("Loading market signals…"):
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # PM Signals panel: from feature_bars we have pm_p, pm_delta_p_1h per bar
                    st.markdown("#### PM signals (probability)")
                    if bars:
                        df = pd.DataFrame(bars)
                        if "ts" in df.columns and "pm_p" in df.columns:
                            df["ts"] = pd.to_datetime(df["ts"])
                            df = df.dropna(subset=["pm_p"]).head(500)
                            if not df.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df["ts"], y=df["pm_p"] * 100, mode="lines", name="P (event) %"))
                                if "pm_delta_p_1h" in df.columns:
                                    fig.add_trace(go.Scatter(x=df["ts"], y=df["pm_delta_p_1h"] * 100, mode="lines", name="Δp 1h (%)"))
                                fig.update_layout(height=280, margin=dict(t=20, b=30), yaxis_title="%", xaxis_title="Time")
                                st.plotly_chart(fig, width="stretch")
                            else:
                                st.info("No PM data in feature_bars for this range.")
                        else:
                            st.info("Missing ts or pm_p in feature_bars.")
                    else:
                        st.info("Load feature bars first.")

                    st.divider()
                    st.markdown("#### News sentiment (recent headlines)")
                    rows = _safe_query(cur, """
                        SELECT ts, source, sentiment_compound, sentiment_model, left(text, 120) as snippet
                        FROM sentiment_scored
                        ORDER BY ts DESC LIMIT 30
                    """)
                    if rows:
                        for r in rows:
                            ts, source, compound, model, snippet = r[0], r[1], r[2] or 0, (r[3] or "—"), (r[4] or "") + "..."
                            color = "positive" if compound > 0.1 else ("negative" if compound < -0.1 else "neutral")
                            st.markdown(f"**{_fmt_ts(ts)}** · {source}")
                            st.caption(f"{snippet}")
                            st.markdown(f'<div class="sentiment-bar {color}" style="width: {min(100, abs(compound)*100)}%;"></div>', unsafe_allow_html=True)
                            st.markdown(f"Compound: **{_fmt_num(compound, 2)}** ({model})")
                            st.markdown("---")
                    else:
                        st.info("No sentiment_scored rows.")

                    st.divider()
                    st.markdown("#### Volatility surface (ATM IV)")
                    if bars:
                        df = pd.DataFrame(bars)
                        if "ts" in df.columns:
                            df["ts"] = pd.to_datetime(df["ts"])
                        fig = go.Figure()
                        for col, label in [("atm_iv_7d", "7d"), ("atm_iv_14d", "14d"), ("atm_iv_30d", "30d")]:
                            if col in df.columns and df[col].notna().any():
                                fig.add_trace(go.Scatter(x=df["ts"], y=df[col] * 100, mode="lines", name=f"ATM IV {label}"))
                        fig.update_layout(height=280, margin=dict(t=20, b=30), yaxis_title="IV %", xaxis_title="Time")
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("Load feature bars to see IV.")
    except Exception as e:
        st.error("Error loading market signals. Check database connection.")
        st.caption(str(e))


# ---------- Page 3: Performance ----------
def page_performance():
    _inject_css()
    st.subheader("Performance")
    underlying = st.sidebar.text_input("Underlying", value="SPY", key="perf_underlying")
    limit_bars = st.sidebar.number_input("Max bars", value=2000, min_value=100, key="perf_bars")
    ablation_path = st.sidebar.text_input("Ablation JSON path", value="ablation_results.json")
    default_start = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    default_end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start = st.date_input("Start date", value=pd.to_datetime(default_start).date(), key="perf_start")
    end = st.date_input("End date", value=pd.to_datetime(default_end).date(), key="perf_end")

    bars = []
    try:
        with st.spinner("Loading feature bars…"):
            from src.envs.options_env import load_feature_bars_from_db
            bars = load_feature_bars_from_db(underlying=underlying, start_date=start.isoformat(), end_date=end.isoformat(), limit=limit_bars)
    except Exception as e:
        st.warning("Could not load feature bars. Run pipeline steps 1–10 and check date range.")
        st.caption(str(e))

    if not bars:
        st.info("Load feature bars (set date range and underlying) to see performance.")
        return

    bars_df = pd.DataFrame(bars)
    if "ts" in bars_df.columns:
        bars_df["ts"] = pd.to_datetime(bars_df["ts"])
    bars = bars_df.to_dict("records")

    try:
        with st.spinner("Evaluating baselines and agent…"):
            from src.envs.options_env import OptionsEnv
            from src.agents.eval import evaluate_policy_with_series
            from src.agents.obs_mask_wrapper import ObsMaskWrapper
            from src.agents.baselines import BuyAndHold
            from src.agents.ablation import SB3PolicyAdapter
            from stable_baselines3 import PPO
            env = OptionsEnv(feature_bars=bars, underlying=underlying)
            policy_bh = BuyAndHold()
            metrics_bh = evaluate_policy_with_series(env, policy_bh, n_episodes=1, seeds=[0])
            model_path = Path(ROOT) / "models" / "ablation_D_ppo_seed0.zip"
            if model_path.exists():
                model = PPO.load(str(model_path))
                env2 = ObsMaskWrapper(OptionsEnv(feature_bars=bars, underlying=underlying), variant="D")
                policy_d = SB3PolicyAdapter(model, algorithm="ppo")
                metrics_d = evaluate_policy_with_series(env2, policy_d, n_episodes=1, seeds=[0])
            else:
                metrics_d = None
    except Exception as e:
        metrics_bh = None
        metrics_d = None
        st.warning("Evaluation failed. If the model is missing, run pipeline step 12 (ablation).")
        st.caption(str(e))

    if metrics_bh and metrics_bh.get("equity_series"):
        eq_bh = metrics_bh["equity_series"]
        ts_list = bars_df["ts"].iloc[1 : len(eq_bh) + 1].tolist() if len(bars_df) >= len(eq_bh) else list(range(len(eq_bh)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_list, y=eq_bh, mode="lines", name="BuyAndHold"))
        if metrics_d and metrics_d.get("equity_series"):
            eq_d = metrics_d["equity_series"]
            fig.add_trace(go.Scatter(x=ts_list[: len(eq_d)], y=eq_d, mode="lines", name="Variant D (Full)"))
        fig.update_layout(title="Equity curve", xaxis_title="Time", yaxis_title="Equity", height=380)
        st.plotly_chart(fig, width="stretch")

    # Ablation comparison table and chart
    ablation_data = None
    ablation_file = (ROOT / ablation_path) if not Path(ablation_path).is_absolute() else Path(ablation_path)
    if ablation_file.exists():
        try:
            from src.agents.analysis import load_ablation_results
            ablation_data = load_ablation_results(str(ablation_file))
        except Exception:
            pass
    if ablation_data:
        agg = [a for a in ablation_data.get("aggregated", []) if a.get("algorithm") == "ppo"]
        if agg:
            variants = [VARIANT_LABELS.get(a["variant"], a["variant"]) for a in agg]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=variants, y=[a["sharpe_mean"] for a in agg], error_y=dict(type="data", array=[a["sharpe_std"] for a in agg]), name="Sharpe"))
            fig.update_layout(title="Sharpe ratio (mean ± std)", xaxis_title="Variant", height=320)
            st.plotly_chart(fig, width="stretch")
            rows = []
            for a in agg:
                rows.append({
                    "Variant": VARIANT_LABELS.get(a["variant"], a["variant"]),
                    "Sharpe": f"{a['sharpe_mean']:.2f} ± {a['sharpe_std']:.2f}",
                    "Max DD": f"{a['max_drawdown_mean']:.4f}",
                    "Hit rate %": f"{a['hit_rate_mean']:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if metrics_d and metrics_d.get("net_delta_series") and metrics_d.get("net_vega_series"):
        delta = metrics_d["net_delta_series"]
        vega = metrics_d["net_vega_series"]
        n = max(len(delta), len(vega))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(n)), y=delta or [0] * n, name="Net delta"))
        fig.add_trace(go.Scatter(x=list(range(n)), y=vega or [0] * n, name="Net vega"))
        fig.update_layout(title="Exposure (Variant D)", height=280)
        st.plotly_chart(fig, width="stretch")


# ---------- Page 4: Trade Impact ----------
def page_trade_impact():
    _inject_css()
    st.subheader("Trade Impact")
    st.caption("Signal → action → outcome for bars where the agent changed position.")
    underlying = st.sidebar.text_input("Underlying", value="SPY", key="impact_underlying")
    limit_bars = st.sidebar.number_input("Max bars", value=800, min_value=100, key="impact_bars")
    show_all_changes = st.sidebar.checkbox("Show all bars with any position change", value=False, key="impact_show_all")

    try:
        from src.envs.options_env import load_feature_bars_from_db, OptionsEnv
        from src.agents.eval import evaluate_policy_with_series
        from src.agents.obs_mask_wrapper import ObsMaskWrapper
        from src.agents.ablation import SB3PolicyAdapter
        from stable_baselines3 import PPO
    except ImportError as e:
        st.error("Import error. Ensure all dependencies are installed.")
        st.caption(str(e))
        return

    try:
        with st.spinner("Loading feature bars…"):
            bars = load_feature_bars_from_db(underlying=underlying, limit=limit_bars)
    except Exception as e:
        st.warning("Could not load feature bars. Run pipeline steps 1–10.")
        st.caption(str(e))
        return

    if not bars or len(bars) < 2:
        st.info("Need feature bars. Run pipeline steps 1–10 and load data.")
        return

    model_path = Path(ROOT) / "models" / "ablation_D_ppo_seed0.zip"
    if not model_path.exists():
        st.info("Model file not found. Run pipeline step 12 (ablation) to generate the Variant D model, then revisit this page.")
        return

    try:
        with st.spinner("Evaluating policy…"):
            model = PPO.load(str(model_path))
            env = ObsMaskWrapper(OptionsEnv(feature_bars=bars, underlying=underlying), variant="D")
            policy = SB3PolicyAdapter(model, algorithm="ppo")
            metrics = evaluate_policy_with_series(env, policy, n_episodes=1, seeds=[0])
    except Exception as e:
        st.error("Evaluation failed. Check that the model file is valid.")
        st.caption(str(e))
        return

    pnl = metrics.get("pnl_series", [])
    delta = metrics.get("net_delta_series", [])
    vega = metrics.get("net_vega_series", [])
    if not delta and not vega:
        st.info("No position series in evaluation output.")
        return

    delta = delta or [0] * len(bars)
    vega = vega or [0] * len(bars)
    pnl = pnl or [0] * len(bars)
    # Find bars where position changed (threshold: 1.0 default, or 0.01 if "show all")
    threshold = 0.01 if show_all_changes else 1.0
    action_bars = []
    for i in range(1, min(len(bars), len(delta), len(vega), len(pnl))):
        d_prev, d_cur = delta[i - 1] or 0, delta[i] or 0
        v_prev, v_cur = vega[i - 1] or 0, vega[i] or 0
        if abs(d_cur - d_prev) >= threshold or abs(v_cur - v_prev) >= threshold:
            bar = bars[i] if i < len(bars) else {}
            ts = bar.get("ts")
            action_bars.append({
                "bar_ix": i,
                "ts": ts,
                "pm_p": bar.get("pm_p"),
                "pm_delta_p_1h": bar.get("pm_delta_p_1h"),
                "sent_news_asset": bar.get("sent_news_asset"),
                "sent_macro_topic": bar.get("sent_macro_topic"),
                "atm_iv_30d": bar.get("atm_iv_30d"),
                "delta_change": (d_cur - d_prev),
                "vega_change": (v_cur - v_prev),
                "pnl_this": pnl[i] if i < len(pnl) else 0,
            })

    if not action_bars:
        st.info("No significant position changes in this run. Try more bars, a different underlying, or enable « Show all bars with any position change » in the sidebar.")
        return

    st.markdown(f"Found **{len(action_bars)}** bars with position changes.")
    for item in action_bars[:20]:
        with st.expander(f"{_fmt_ts(item['ts'])} — Δdelta={item['delta_change']:.0f}, Δvega={item['vega_change']:.0f} · PnL={_fmt_num(item['pnl_this'])}"):
            st.markdown("**Signals at bar**")
            st.write(f"PM probability: {_fmt_pct(item['pm_p'])} · Δp 1h: {_fmt_pct(item['pm_delta_p_1h'])}")
            st.write(f"News sentiment: {_fmt_num(item['sent_news_asset'])} · Macro: {_fmt_num(item['sent_macro_topic'])}")
            st.write(f"ATM IV 30d: {_fmt_pct(item['atm_iv_30d'])}")
            st.markdown("**Outcome**")
            st.write(f"PnL this bar: **{_fmt_num(item['pnl_this'])}**")


def main():
    st.set_page_config(page_title="Options Agent Dashboard", layout="wide")
    st.sidebar.title("Options Agent Dashboard")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Market Signals", "Performance", "Trade Impact"],
    )
    if page == "Overview":
        page_overview()
    elif page == "Market Signals":
        page_market_signals()
    elif page == "Performance":
        page_performance()
    else:
        page_trade_impact()


if __name__ == "__main__":
    main()
