"""
FastAPI app for the Options Agent web dashboard.
Serves /api/* JSON and static frontend at /. Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[2]

MONITOR_TABLES = [
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


def _serialize_ts(ts) -> str | None:
    if ts is None:
        return None
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    yield


app = FastAPI(title="Options Agent Dashboard API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend (set at startup or when building Docker)
STATIC_DIR = Path(os.environ.get("DASHBOARD_STATIC_DIR", str(ROOT / "web" / "out" / "browser")))


@app.get("/health")
def health():
    """Cloud Run health check; verifies DB connectivity."""
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/overview")
def api_overview():
    """KPIs, pipeline status, feature coverage, last pipeline run."""
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                total_bars = _safe_query(cur, "SELECT count(*) FROM feature_bars")
                total_prices = _safe_query(cur, "SELECT count(*) FROM pm_prices")
                total_sentiment = _safe_query(cur, "SELECT count(*) FROM sentiment_scored")
                total_events = _safe_query(cur, "SELECT count(*) FROM pm_events")
                res_latest = _safe_query(cur, "SELECT max(ts) FROM feature_bars")
                n_bars = total_bars[0][0] if total_bars else 0
                n_prices = total_prices[0][0] if total_prices else 0
                n_sentiment = total_sentiment[0][0] if total_sentiment else 0
                n_events = total_events[0][0] if total_events else 0
                latest_ts = res_latest[0][0] if res_latest and res_latest[0][0] else None

                now = datetime.now(timezone.utc)
                cutoff_24h = now - timedelta(hours=24)
                statuses = []
                for table_name, ts_kind, ts_col in MONITOR_TABLES:
                    total_rows = _safe_query(cur, f"SELECT count(*) FROM {table_name}")
                    total = total_rows[0][0] if total_rows else 0
                    recent = 0
                    latest_ts_t = None
                    if ts_col:
                        if ts_kind == "ts":
                            res = _safe_query(cur, f"SELECT count(*), max({ts_col}) FROM {table_name} WHERE {ts_col} >= %s", (cutoff_24h,))
                        else:
                            res = _safe_query(cur, f"SELECT count(*), max({ts_col}) FROM {table_name} WHERE {ts_col} >= %s", (cutoff_24h.date(),))
                        if res:
                            recent, latest_ts_t = res[0][0], res[0][1]
                    if total == 0:
                        status = "empty"
                    elif recent > 0:
                        status = "fresh"
                    elif latest_ts_t and hasattr(latest_ts_t, "date"):
                        age = (now.date() - latest_ts_t.date()).days
                        status = "stale" if age <= 2 else "old"
                    else:
                        status = "old"
                    statuses.append({"table": table_name, "status": status})

                cutoff_7d = (now - timedelta(days=7)).date()
                cov_rows = _safe_query(cur, """
                    SELECT count(*), count(atm_iv_7d), count(pm_p), count(sent_news_asset), count(equity_return_1d)
                    FROM feature_bars WHERE ts::date >= %s
                """, (cutoff_7d,))
                coverage = None
                if cov_rows and cov_rows[0][0] > 0:
                    total, iv, pm, sent, eq = cov_rows[0]
                    coverage = {"bars": total, "iv": iv, "pm": pm, "sentiment": sent, "equity": eq}

                last_line = None
                sentiment_summary = None
                try:
                    cur.execute("SELECT last_line, sentiment_summary FROM dashboard_pipeline_log WHERE id = 1")
                    row = cur.fetchone()
                    if row:
                        last_line, sentiment_summary = row[0], row[1]
                except Exception:
                    pass
                if sentiment_summary is None:
                    model_rows = _safe_query(cur, "SELECT sentiment_model, count(*) FROM sentiment_scored GROUP BY sentiment_model")
                    sentiment_summary = ", ".join(f"{m or 'unknown'}: {c}" for m, c in model_rows) if model_rows else "—"

                return {
                    "kpis": {
                        "feature_bars": n_bars,
                        "pm_prices": n_prices,
                        "pm_events": n_events,
                        "sentiment_scored": n_sentiment,
                        "latest_ts": _serialize_ts(latest_ts),
                    },
                    "pipeline_status": statuses,
                    "feature_coverage_7d": coverage,
                    "last_pipeline_line": last_line,
                    "sentiment_summary": sentiment_summary,
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
def api_signals(
    underlying: str = Query("SPY", description="Underlying symbol"),
    limit: int = Query(500, ge=50, le=5000),
):
    """Feature bars (PM + IV) and recent sentiment headlines."""
    try:
        from src.envs.options_env import load_feature_bars_from_db
        from src.db import get_connection
        bars = load_feature_bars_from_db(underlying=underlying, limit=limit)
        bars_out = []
        for b in bars:
            row = dict(b)
            row["ts"] = _serialize_ts(row.get("ts"))
            bars_out.append(row)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ts, source, sentiment_compound, sentiment_model, left(text, 120) as snippet
                    FROM sentiment_scored ORDER BY ts DESC LIMIT 30
                """)
                rows = cur.fetchall()
        headlines = [
            {
                "ts": _serialize_ts(r[0]),
                "source": r[1],
                "compound": float(r[2] or 0),
                "model": r[3] or "—",
                "snippet": (r[4] or "") + ("..." if r[4] else ""),
            }
            for r in rows
        ]
        return {"feature_bars": bars_out, "headlines": headlines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
def api_performance(
    underlying: str = Query("SPY", description="Underlying symbol"),
):
    """Precomputed equity series, ablation summary, exposure (from dashboard_evaluation + dashboard_ablation)."""
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ts_index, equity_bh, equity_d, metrics_bh, metrics_d, exposure_delta, exposure_vega FROM dashboard_evaluation WHERE underlying = %s",
                    (underlying,),
                )
                row = cur.fetchone()
        if not row:
            return {"equity": None, "ablation": None, "exposure": None}

        ts_index, equity_bh, equity_d, metrics_bh, metrics_d, exposure_delta, exposure_vega = row
        equity = {
            "ts": ts_index or [],
            "buy_and_hold": equity_bh or [],
            "variant_d": equity_d or [],
        }
        exposure = {
            "delta": exposure_delta or [],
            "vega": exposure_vega or [],
        }
        ablation = None
        try:
            from src.db import get_connection
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT aggregated_json FROM dashboard_ablation WHERE id = 1")
                    r = cur.fetchone()
                    if r and r[0]:
                        ablation = r[0]
        except Exception:
            pass
        return {"equity": equity, "metrics_bh": metrics_bh, "metrics_d": metrics_d, "ablation": ablation, "exposure": exposure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-impact")
def api_trade_impact(
    underlying: str = Query("SPY"),
    limit: int = Query(200, ge=1, le=500),
    show_all: bool = Query(False, description="Include bars with any position change (else threshold 0.01)"),
):
    """Precomputed trade-impact bars (signal → action → PnL)."""
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT bars FROM dashboard_trade_impact WHERE underlying = %s", (underlying,))
                row = cur.fetchone()
        if not row or not row[0]:
            return {"bars": []}
        bars_raw = row[0]
        threshold = 0.001 if show_all else 0.01
        filtered = [b for b in bars_raw if abs(b.get("delta_change", 0)) >= threshold or abs(b.get("vega_change", 0)) >= threshold]
        # Convert to SignalFeedEntry format for frontend compatibility
        feed = []
        for b in filtered[:limit]:
            feed.append({
                "ts": b.get("ts"),
                "underlying": underlying,
                "news": b.get("headlines") or [],
                "pm_signals": [],
                "action": {"delta_change": b.get("delta_change"), "vega_change": b.get("vega_change")},
                "outcome": {"pnl": b.get("pnl_this")},
                "iv": {
                    "atm_iv_7d": None,
                    "atm_iv_14d": None,
                    "atm_iv_30d": b.get("atm_iv_30d"),
                },
            })
        return {"bars": feed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_signal_feed_mapping() -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Build underlying -> token_ids and token_id -> {event_name, platform, affected_underlyings}."""
    import yaml
    config_dir = ROOT / "configs"
    mapping_path = config_dir / "mapping.yaml"
    underlying_to_tokens: dict[str, list[str]] = {}
    token_to_event: dict[str, dict] = {}
    if not mapping_path.exists():
        return underlying_to_tokens, token_to_event
    with open(mapping_path) as f:
        mapping = yaml.safe_load(f) or []
    if not isinstance(mapping, list):
        return underlying_to_tokens, token_to_event
    for entry in mapping:
        tokens = entry.get("token_ids") or {}
        underlyings = entry.get("affected_underlyings") or []
        event_slug = entry.get("event_slug") or ""
        for tid in tokens.values():
            if not (tid and str(tid).strip()):
                continue
            tid = str(tid).strip()
            for u in underlyings:
                u = str(u).strip()
                if u not in underlying_to_tokens:
                    underlying_to_tokens[u] = []
                if tid not in underlying_to_tokens[u]:
                    underlying_to_tokens[u].append(tid)
            if tid not in token_to_event:
                token_to_event[tid] = {
                    "event_name": event_slug.replace("-", " ").title(),
                    "platform": "",
                    "affected_underlyings": list(underlyings),
                }
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                for tid in token_to_event:
                    cur.execute(
                        "SELECT m.platform, e.title FROM pm_markets m JOIN pm_events e ON e.event_id = m.event_id WHERE %s = ANY(m.token_ids) LIMIT 1",
                        (tid,),
                    )
                    row = cur.fetchone()
                    if row:
                        token_to_event[tid]["platform"] = row[0] or ""
                        if row[1]:
                            token_to_event[tid]["event_name"] = row[1]
    except Exception:
        pass
    return underlying_to_tokens, token_to_event


@app.get("/api/signal-feed")
def api_signal_feed(
    underlying: str = Query("SPY"),
    limit: int = Query(100, ge=1, le=500),
    position_changes_only: bool = Query(False, description="Only bars with position changes"),
):
    """Chronological signal feed: Source -> Stock -> Position. News, PM events, action, outcome."""
    try:
        from src.db import get_connection
        from src.envs.options_env import load_feature_bars_from_db
        underlying_to_tokens, token_to_event = _load_signal_feed_mapping()
        token_ids = underlying_to_tokens.get(underlying, [])

        if position_changes_only:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT bars FROM dashboard_trade_impact WHERE underlying = %s", (underlying,))
                    row = cur.fetchone()
            bars_raw = (row[0] if row and row[0] else [])[:limit]
            bars = []
            for b in bars_raw:
                bars.append({
                    "ts": b.get("ts"),
                    "pm_p": b.get("pm_p"),
                    "pm_delta_p_1h": b.get("pm_delta_p_1h"),
                    "sent_news_asset": b.get("sent_news_asset"),
                    "atm_iv_7d": None,
                    "atm_iv_14d": None,
                    "atm_iv_30d": b.get("atm_iv_30d"),
                    "delta_change": b.get("delta_change"),
                    "vega_change": b.get("vega_change"),
                    "pnl_this": b.get("pnl_this"),
                    "headlines": b.get("headlines") or [],
                })
        else:
            # Pull a wider window and keep the most recent bars for UI freshness.
            recent_limit = max(limit * 20, 500)
            feature_bars = load_feature_bars_from_db(underlying=underlying, limit=recent_limit)
            if len(feature_bars) > limit:
                feature_bars = feature_bars[-limit:]
            bars = []
            for b in feature_bars:
                bars.append({
                    "ts": b.get("ts"),
                    "pm_p": b.get("pm_p"),
                    "pm_delta_p_1h": b.get("pm_delta_p_1h"),
                    "sent_news_asset": b.get("sent_news_asset"),
                    "atm_iv_7d": b.get("atm_iv_7d"),
                    "atm_iv_14d": b.get("atm_iv_14d"),
                    "atm_iv_30d": b.get("atm_iv_30d"),
                    "delta_change": None,
                    "vega_change": None,
                    "pnl_this": None,
                    "headlines": None,
                })

        # --- Batch-load sentiment headlines for all bars in one query ---
        from dateutil import parser as date_parser
        bar_dts = []
        for bar in bars:
            ts = bar.get("ts")
            if not ts:
                bar_dts.append(None)
                continue
            try:
                bar_dts.append(date_parser.isoparse(ts.replace("Z", "+00:00")) if isinstance(ts, str) else ts)
            except Exception:
                bar_dts.append(None)

        # Find the full time range across all bars for one batch sentiment query
        valid_dts = [d for d in bar_dts if d is not None]
        all_headlines: list[dict] = []
        if valid_dts:
            t_min = min(valid_dts) - timedelta(hours=24)
            t_max = max(valid_dts)
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT ts, source, left(text, 400), sentiment_compound, sentiment_model
                        FROM sentiment_scored WHERE ts >= %s AND ts <= %s ORDER BY ts DESC LIMIT 2000
                        """,
                        (t_min, t_max),
                    )
                    all_headlines = [
                        {
                            "ts": r[0],
                            "source": r[1] or "",
                            "text": (r[2] or "").strip(),
                            "compound": float(r[3]) if r[3] is not None else 0.0,
                            "model": r[4] or "",
                        }
                        for r in cur.fetchall()
                    ]

        # Batch-load latest PM signals (one query, not per-bar)
        pm_latest: list[dict] = []
        if valid_dts:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if token_ids:
                        placeholders = ", ".join(["%s"] * len(token_ids))
                        cur.execute(
                            f"""
                            SELECT DISTINCT ON (token_id) token_id, p, delta_p_1h, ts
                            FROM pm_features
                            WHERE token_id IN ({placeholders}) AND ts <= %s
                            ORDER BY token_id, ts DESC
                            """,
                            (*token_ids, max(valid_dts)),
                        )
                        for tid, p, d1h, _ in cur.fetchall():
                            if p is not None:
                                ev = token_to_event.get(tid, {"event_name": str(tid)[:20], "platform": "?", "affected_underlyings": [underlying]})
                                pm_latest.append({
                                    "event_name": ev.get("event_name", ""),
                                    "platform": ev.get("platform", ""),
                                    "token_id": tid,
                                    "probability": float(p),
                                    "delta_1h": float(d1h) if d1h is not None else None,
                                    "affected_underlyings": ev.get("affected_underlyings", [underlying]),
                                })
                    if not pm_latest:
                        cur.execute(
                            """
                            SELECT DISTINCT ON (pf.token_id) pf.token_id, pf.p, pf.delta_p_1h, m.platform, e.title
                            FROM pm_features pf
                            LEFT JOIN pm_markets m ON pf.token_id = ANY(m.token_ids)
                            LEFT JOIN pm_events e ON e.event_id = m.event_id
                            WHERE pf.ts <= %s
                            ORDER BY pf.token_id, pf.ts DESC
                            LIMIT 5
                            """,
                            (max(valid_dts),),
                        )
                        for tid, p, d1h, platform, title in cur.fetchall():
                            if tid and p is not None:
                                pm_latest.append({
                                    "event_name": title or str(tid)[:20],
                                    "platform": platform or "PM",
                                    "token_id": str(tid),
                                    "probability": float(p),
                                    "delta_1h": float(d1h) if d1h is not None else None,
                                    "affected_underlyings": [underlying],
                                })

        # --- Build feed entries using batched data ---
        feed = []
        for i, bar in enumerate(bars):
            ts = bar.get("ts")
            bar_dt = bar_dts[i] if i < len(bar_dts) else None
            if not ts:
                continue

            # Assign headlines from pre-loaded batch
            news = bar.get("headlines")
            if news is None and bar_dt and all_headlines:
                window_start = bar_dt - timedelta(hours=24)
                news = []
                for h in all_headlines:
                    h_ts = h["ts"]
                    if h_ts and window_start <= h_ts <= bar_dt:
                        news.append({
                            "ts": _serialize_ts(h_ts),
                            "source": h["source"],
                            "text": h["text"],
                            "compound": h["compound"],
                            "model": h["model"],
                        })
                        if len(news) >= 5:
                            break
            else:
                news = news or []

            feed.append({
                "ts": _serialize_ts(ts),
                "underlying": underlying,
                "news": news,
                "pm_signals": pm_latest,
                "action": {"delta_change": bar.get("delta_change"), "vega_change": bar.get("vega_change")} if bar.get("delta_change") is not None else None,
                "outcome": {"pnl": bar.get("pnl_this")} if bar.get("pnl_this") is not None else None,
                "iv": {
                    "atm_iv_7d": bar.get("atm_iv_7d"),
                    "atm_iv_14d": bar.get("atm_iv_14d"),
                    "atm_iv_30d": bar.get("atm_iv_30d"),
                },
            })
        return {"feed": feed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ablation")
def api_ablation():
    """Aggregated ablation metrics for Performance page chart/table."""
    try:
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT aggregated_json FROM dashboard_ablation WHERE id = 1")
                row = cur.fetchone()
        if not row or not row[0]:
            return {"aggregated": []}
        return {"aggregated": row[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Task monitoring endpoints
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _PydanticBaseModel


class _LaunchTaskRequest(_PydanticBaseModel):
    task_type: str  # 'ablation', 'pipeline', 'reports', 'snapshot'
    config: dict[str, Any] = {}
    label: str | None = None


def _serialize_task(t: dict) -> dict:
    """Make a task_runs row JSON-serializable and add elapsed/ETA."""
    out = {}
    for k, v in t.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    if t.get("started_at"):
        end = t.get("completed_at") or datetime.now(timezone.utc)
        started = t["started_at"]
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if hasattr(end, "tzinfo") and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        elapsed = (end - started).total_seconds()
        out["elapsed_seconds"] = round(elapsed, 1)
        pct = t.get("progress_pct") or 0
        if 0 < pct < 100 and elapsed > 0:
            out["eta_seconds"] = round(elapsed * (100 - pct) / pct, 1)
        else:
            out["eta_seconds"] = None
    else:
        out["elapsed_seconds"] = None
        out["eta_seconds"] = None
    return out


@app.get("/api/tasks")
def api_list_tasks(limit: int = Query(20, ge=1, le=100), status: str | None = None):
    from src.api.tasks import list_tasks
    tasks = list_tasks(limit=limit, status=status)
    return {"tasks": [_serialize_task(t) for t in tasks]}


@app.get("/api/tasks/{task_id}")
def api_get_task(task_id: int):
    from src.api.tasks import get_task
    t = get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return _serialize_task(t)


@app.post("/api/tasks")
def api_launch_task(req: _LaunchTaskRequest):
    from src.api.tasks import create_task, has_running_task, launch_task
    if has_running_task(req.task_type):
        raise HTTPException(status_code=409, detail=f"A '{req.task_type}' task is already running")
    task_id = create_task(req.task_type, req.config, label=req.label)
    pid = launch_task(task_id)
    return {"task_id": task_id, "pid": pid, "status": "running"}


@app.post("/api/tasks/{task_id}/cancel")
def api_cancel_task(task_id: int):
    from src.api.tasks import cancel_task
    ok = cancel_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Task not running or already finished")
    return {"task_id": task_id, "status": "cancelled"}


# Serve static frontend (Angular SPA: out/browser/index.html + hashed JS/CSS assets)
def _serve_frontend(path: str = ""):
    path = path.rstrip("/")
    # Serve actual files (JS, CSS, images, fonts, etc.)
    file_path = (STATIC_DIR / path) if path else (STATIC_DIR / "index.html")
    if file_path.is_file():
        return FileResponse(file_path)
    # For SPA routes, always return index.html (Angular Router handles routing)
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend not built. Run: cd web && npm run build")


@app.get("/")
def serve_root():
    return _serve_frontend("")


@app.get("/{path:path}")
def serve_static(path: str):
    if path.startswith("api/") or path == "health":
        raise HTTPException(status_code=404, detail="Not found")
    return _serve_frontend(path)
