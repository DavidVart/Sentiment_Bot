# AI Options Trading Agent: Reinforcement Learning with Prediction Markets and Sentiment

---

## 1. Project title and research question

**Title:** AI Options Trading Agent: combining prediction markets, news/social sentiment, and options features for reinforcement learning-based volatility and delta management.

**Research question:** Can an RL agent that observes (i) options surface and underlying price features, (ii) news and social sentiment, and (iii) prediction-market probabilities improve risk-adjusted performance (e.g. Sharpe ratio, drawdown) over baseline strategies, and which observation subsets contribute most (ablation)?

---

## 2. Architecture (text-based)

```
                    +------------------+
                    |  Polymarket /    |
                    |  Kalshi APIs     |
                    +--------+---------+
                             | pm_events, pm_markets, pm_prices
                             v
+------------------+   +------------------+   +------------------+
|  NewsAPI /       |   |  Polygon /       |   |  pm_feature_     |
|  Reddit (PRAW)   |   |  Tradier /       |   |  builder         |
+--------+---------+   |  yfinance        |   +--------+---------+
         |             +--------+---------+            |
         | sentiment_docs       | equity_bars,         | pm_features
         v                      | options_snapshots   v
+------------------+            v                +------------------+
|  sentiment_      |   +------------------+     |                  |
|  writer          |   |  options_        |     |   align.py       |
|  score_docs      |   |  features        |     |   (build_row,     |
+--------+---------+   +--------+---------+     |    run_build_    |
         |                      |               |    feature_      |
         | sentiment_scored     | options_      |    matrix)       |
         v                      | features      |                  |
+------------------+            v                +--------+---------+
|  sentiment_      |   +------------------+              |
|  features        |   |  feature_bars    | <-----------+  (15-min bars:
+--------+---------+   |  (master matrix) |     JOIN      options + sentiment
         |             +--------+---------+              + PM + equity)
         +---------------------+                         |
                               v                         v
                    +------------------+     +------------------+
                    |  OptionsEnv      |     |  Baselines /     |
                    |  (Gymnasium)     |     |  PPO-SAC agents  |
                    |  obs(52),        |     |  (ablation       |
                    |  action(4)       |     |   variants A-D)  |
                    +--------+---------+     +--------+---------+
                             |                         |
                             v                         v
                    +------------------+     +------------------+
                    |  eval.py         |     |  analysis.py     |
                    |  (metrics,       |     |  (tables, plots, |
                    |   regime_split,  |     |   case studies,  |
                    |   bootstrap)     |     |   reports)       |
                    +------------------+     +------------------+
```

**Data flow (summary):** Raw data (PM, sentiment, equity, options) is ingested into PostgreSQL, then turned into per-asset/time features. The align module joins them onto a 15-min market-hours clock into `feature_bars`. The RL environment reads `feature_bars`, exposes a 52-dimensional observation and a `MultiDiscrete([3,3,3,3])` action space (81 combinations across direction, strike, expiry, and size). Options are priced with Black-Scholes and the execution simulator applies bid-ask spreads (0.5-3%), per-contract fees, slippage, and a 1-bar delay. PPO/SAC agents are trained and evaluated; the evaluation harness and analysis module produce metrics, regime splits, and thesis outputs (tables, plots, case studies).

### Key results (200K in-sample, PPO)

| Variant | Observation dims | Sharpe ratio (mean +/- std) |
|---------|------------------|-----------------------------|
| A (Base) | 45 | 10.61 +/- 4.70 |
| B (+Sentiment) | 48 | 7.67 +/- 2.33 |
| **C (+Prediction Markets)** | **49** | **14.69 +/- 4.51** |
| D (Full) | 52 | 10.24 +/- 1.55 |

Variant C outperforms all others by 38.5% over the base case. VADER sentiment at daily frequency (Variant B) hurts performance. Walk-forward cross-validation preserves the ranking C > A > B > D. All trained agents beat the Delta-Neutral (4.25), Buy-and-Hold (-2.61), and Random Policy (-4.40) baselines.

---

## 3. Setup instructions

### Python and dependencies

- **Python:** 3.10 or newer.
- **Install (editable, with dev extras):**

```bash
cd /path/to/Sentiment_Bot
pip install -e ".[dev]"
```

**If you see** `setuptools is not available in the build environment` **(e.g. when installing `yfinance`’s dependency `multitasking` or optional `py_vollib`):** upgrade pip/setuptools, then install the project using the helper script so legacy packages are built with your current environment:

```bash
python -m pip install --upgrade pip setuptools wheel
bash scripts/install_with_legacy_deps.sh
```

Core dependencies (see `pyproject.toml`) include: `gymnasium`, `numpy`, `pandas`, `psycopg2-binary`, `pyyaml`, `python-dotenv`, `stable-baselines3`, `scipy`, `matplotlib`, `streamlit`, `plotly`, and connectors (e.g. `py-clob-client`, `requests`, `httpx`) for Polymarket/Kalshi and market data.

### Sentiment analysis (AI)

Sentiment scoring uses **FinBERT** (ProsusAI/finbert) for news (e.g. NewsAPI) and **VADER** for other sources (e.g. Reddit). FinBERT is loaded with a timeout; if it does not load in time (e.g. 120s), the pipeline falls back to VADER for all documents. The dashboard shows which model scored each headline (Market Signals) and a summary by model on the Overview page. See `src/connectors/sentiment/scoring.py` and pipeline step 6 (`score_sentiment`).

### Environment variables

Use a `.env` file in the project root (do not commit secrets). Example:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Full Postgres URL, e.g. `postgresql://user:pass@host:5432/dbname` |
| Or: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Alternative to `DATABASE_URL` |
| `MASSIVE_API` | Polygon.io API key (equity/options); if unset, equity backfill uses yfinance |
| `TRADIER_API_TOKEN` | (Optional) Tradier API token for options chains with Greeks |
| `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_API_KEY` | (Optional) Polymarket CLOB/auth |
| `KALSHI_API_KEY`, `KALSHI_MEMBER_ID` | (Optional) Kalshi API |
| NewsAPI / Reddit | As required by `backfill_sentiment` (see config/connectors) |

### PostgreSQL

1. Install and start PostgreSQL (e.g. 14+).
2. Create a database, e.g. `createdb sentiment_bot`.
3. Set `DATABASE_URL` or `POSTGRES_*` so the app can connect.

### Migrations

Apply all migrations (creates/updates tables for PM, equity, options, sentiment, feature_bars):

```bash
python -c "from src.db import apply_migrations; apply_migrations()"
```

Or run the full pipeline once (see below), which runs migrations in step 1.

---

## 4. Running each pipeline step individually

| Step | Command / usage |
|------|------------------|
| **1. Migrations** | `python -c "from src.db import apply_migrations; apply_migrations()"` |
| **2. Equity backfill** | `python scripts/backfill_equity.py [--years 2] [--symbols SPY QQQ]` |
| **3. Options backfill** | `python scripts/backfill_options.py [--symbols SPY] [--date YYYY-MM-DD] [--tradier]` |
| **4. PM backfill** | `python -m scripts.backfill [--no-polymarket] [--no-kalshi] [--max-events 25] [--days 365]` |
| **5. Sentiment backfill** | `python scripts/backfill_sentiment.py [--days 7] [--source news|reddit|all] [--migrate]` |
| **6. Score sentiment** | `python scripts/score_sentiment.py [--model auto|vader|finbert] [--migrate]` |
| **7. PM features** | `python scripts/build_pm_features.py [--token-id ID] [--schema-version 1]` |
| **8. Options features** | `python scripts/build_options_features.py [--underlying SPY] [--date YYYY-MM-DD]` |
| **9. Sentiment features** | `python scripts/build_sentiment_features.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--migrate]` |
| **10. Feature matrix** | `python scripts/build_feature_matrix.py [--underlying SPY] [--start-date ...] [--end-date ...] [--migrate]` |
| **11. Baselines** | `python scripts/run_baselines.py [--underlying SPY] [--limit 2000] [--n_episodes 3] [--out results.json]` |
| **12. Ablation** | See section 6. |
| **13. Reports** | See section 7. |

Symbols/defaults often come from `configs/universe.yaml` and `configs/mapping.yaml`.

---

## 5. Full pipeline (integration test)

Single entry point for the whole pipeline (migrations → backfills → feature builds → baselines → ablation smoke test → reports):

```bash
python scripts/run_full_pipeline.py [options]
```

**Options:**

- `--dry-run`: print which steps would run; do not execute.
- `--steps 1,2,10`: run only the listed step numbers (comma-separated).
- `--equity-years 2`: years of history for equity backfill (default 2).
- `--sentiment-days 7`: days of history for sentiment backfill (default 7).

**Examples:**

```bash
# See planned steps only
python scripts/run_full_pipeline.py --dry-run

# Run only migrations and equity backfill
python scripts/run_full_pipeline.py --steps 1,2

# Run full pipeline (may require DB and API keys)
python scripts/run_full_pipeline.py --equity-years 2
```

If a step fails, the script logs the error and continues with the next step. At the end it prints which steps succeeded/failed and row counts for each DB table.

---

## 6. Ablation study

Ablation trains and evaluates four observation variants (A: base, B: +sentiment, C: +PM, D: full) with PPO and/or SAC over multiple seeds.

**Run full ablation (PPO + SAC, 5 seeds, 50k timesteps):**

```bash
python scripts/run_ablation.py --algorithm both --seeds 5 --timesteps 50000 --out-json ablation_results.json --out-csv ablation_results.csv
```

**Run PPO only, 1 seed, short smoke test:**

```bash
python scripts/run_ablation.py --algorithm ppo --seeds 1 --timesteps 1000 --out-json ablation_results.json
```

Requires `feature_bars` in PostgreSQL (run steps 1-10 first). Models are saved under `models/` (e.g. `ablation_D_ppo_seed0.zip`). Results are written to the given JSON/CSV and include metrics and p-values vs variant A.

---

## 7. Generating reports

Thesis outputs (tables, plots, regime breakdown, event case studies) from ablation results and optional feature bars:

```bash
python scripts/generate_reports.py --ablation-json ablation_results.json [--output-dir reports] [--models-dir models] [--feature-bars-db 1] [--limit 2000]
```

- **Tables:** Ablation table (mean ± std, p-values) and regime table (low/medium/high VIX) as CSV and Markdown under `--output-dir`.
- **Plots:** Equity curves, drawdown, rolling Sharpe, exposure (Variant D), and (if models exist) event case study figures.
- Use `--feature-bars-db` to load bars from DB for plots; otherwise synthetic bars are used where needed.

---

## 8. Dashboard

**Web dashboard (primary):** FastAPI backend + Angular 17 frontend. Five pages: Overview, Market Signals, Performance, Trade Impact, and Tasks. Data is read from PostgreSQL and from precomputed snapshot tables (filled by the daily pipeline + snapshot script).

**Run locally:**

```bash
cd web && npm install && npm run build && cd ..
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
# Open http://localhost:8080
```

Set `DASHBOARD_STATIC_DIR=web/out/browser` if needed (default when running from project root). The API serves the built Angular export from that directory.

**Optional, Streamlit (legacy):** For a quick local UI without building the frontend: `streamlit run src/dashboard/app.py`

### Running the dashboard in Docker

The Docker image builds the Angular frontend and runs the **FastAPI** app (no Streamlit). The pipeline runs **separately** (e.g. daily via Cloud Run Job or cron). After the pipeline, run `scripts/write_dashboard_snapshot.py` so the web dashboard has precomputed performance and trade-impact data.

**Build and run:**

```bash
docker build -t options-agent-web .
docker run -p 8080:8080 -e DATABASE_URL=postgresql://user:pass@host:5432/dbname options-agent-web
# Dashboard: http://localhost:8080
```

**Local dev with Postgres:** Use `docker compose` (see `docker-compose.yml`). Update the compose file to use the new image and port 8080 if needed.

**Hosting on a website and automating daily updates:** To have the dashboard always available at a URL and the pipeline run every day without manual steps, use **GCP** (recommended) or another cloud:

- **GCP (recommended):** Deploy the web app as a **Cloud Run** service (always-on URL) and run the pipeline + **dashboard snapshot** as a **Cloud Run Job** triggered by **Cloud Scheduler**. Use **Cloud SQL** for PostgreSQL. Step-by-step: **[docs/deploy-gcp.md](docs/deploy-gcp.md)**. Add `DATABASE_URL` and API keys to Secret Manager (or `.env` for local).
- **Alternatives:** [Render](https://render.com) (Web Service + Cron Job), **Railway**, or **Fly.io** (Web Service + Cron Job for pipeline and snapshot).
- **Host cron:**  
  `30 17 * * 1-5 cd /path/to/Sentiment_Bot && .venv/bin/python scripts/run_full_pipeline.py --steps 2,3,5,6,7,8,9,10 && .venv/bin/python scripts/write_dashboard_snapshot.py >> pipeline_cron.log 2>&1`

The dashboard reads from PostgreSQL and from `dashboard_*` snapshot tables; run the pipeline and snapshot script on a schedule so the site shows fresh data.

---

## 9. Tests

Run the full test suite:

```bash
python -m pytest tests/ -v --tb=short
```

With coverage:

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover connectors, DB writer, feature builders, align, options env, baselines, eval, ablation, analysis, report generation, and the full-pipeline script (e.g. dry-run and selected steps).

---

## 10. Repository structure

| Path | Description |
|------|-------------|
| **configs/** | YAML: data_sources, mapping (PM → underlyings/tokens), universe (symbols). |
| **migrations/** | SQL migrations for PM, equity_bars, options_snapshots, options_features, sentiment_docs, sentiment_scored, sentiment_features, feature_bars. |
| **src/agents/** | **baselines.py**: BuyAndHold, FixedLongVol, SimpleEventRule, DeltaNeutral, RandomPolicy. **eval.py**: evaluate_policy, compute_metrics, regime_split, bootstrap_sharpe, walk_forward_evaluate. **train_sb3.py**: train_agent (PPO/SAC), split_bars_by_time, DiscreteToBoxWrapper. **ablation.py**: run_ablation, save_ablation_results, SB3PolicyAdapter. **analysis.py**: load_ablation_results, format_ablation_table, format_regime_table, plot_*, generate_event_case_study. **obs_mask_wrapper.py**: ObsMaskWrapper and variant masks for ablation. |
| **src/connectors/** | Polymarket (Gamma, CLOB), Kalshi API, market data (Polygon, yfinance, Tradier), sentiment (NewsAPI, Reddit). |
| **src/api/** | **main.py**: FastAPI app (Overview, Signals, Performance, Trade Impact, Tasks JSON API; serves built web dashboard). |
| **src/dashboard/** | **app.py**: Streamlit app (legacy local UI). |
| **web/** | Angular 17 frontend (Overview, Market Signals, Performance, Trade Impact, Tasks); build output in `web/out/browser`. |
| **src/db.py** | PostgreSQL connection and apply_migrations. |
| **src/envs/** | **options_env.py**: OptionsEnv (Gymnasium), load_feature_bars_from_db. **portfolio_constructor.py**: build_target_positions (vega/delta to legs). **execution_sim.py**: ExecutionSimulator. **reward.py**, **constraints.py**: reward and risk logic. |
| **src/features/** | **align.py**: build_row, run_build_feature_matrix (master feature_bars). **pm_feature_builder.py**: PM features from pm_prices. **options_features.py**: ATM IV, skew, realized vol from snapshots. **sentiment_features.py**: 15-min sentiment bars. |
| **src/ingestion/** | Backfill and write logic for PM, equity, options, sentiment (pm_writer, sentiment_writer, backfill_*.py). |
| **src/utils/** | Schemas (pydantic), logging, HTTP retry. |
| **scripts/** | CLI entrypoints: backfill, backfill_equity, backfill_options, backfill_sentiment, score_sentiment, build_pm_features, build_options_features, build_sentiment_features, build_feature_matrix, run_baselines, run_ablation, generate_reports, **run_full_pipeline.py** (end-to-end). |

---

## 11. License

MIT License. See [LICENSE](LICENSE) for details.
