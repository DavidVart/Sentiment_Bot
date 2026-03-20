-- Dashboard snapshot tables: precomputed by daily job; API reads only (no model at request time)

-- Single row: last pipeline log line and sentiment model summary
CREATE TABLE IF NOT EXISTS dashboard_pipeline_log (
    id INT PRIMARY KEY DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_line TEXT,
    sentiment_summary TEXT,
    CONSTRAINT dashboard_pipeline_log_single_row CHECK (id = 1)
);

-- Single row: aggregated ablation results (from ablation_results.json)
CREATE TABLE IF NOT EXISTS dashboard_ablation (
    id INT PRIMARY KEY DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    aggregated_json JSONB,
    CONSTRAINT dashboard_ablation_single_row CHECK (id = 1)
);

-- One row per underlying: precomputed evaluation (equity series, metrics, exposure)
CREATE TABLE IF NOT EXISTS dashboard_evaluation (
    underlying TEXT NOT NULL PRIMARY KEY,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_index JSONB,
    equity_bh JSONB,
    equity_d JSONB,
    metrics_bh JSONB,
    metrics_d JSONB,
    exposure_delta JSONB,
    exposure_vega JSONB
);

-- One row per underlying: precomputed trade-impact bars
CREATE TABLE IF NOT EXISTS dashboard_trade_impact (
    underlying TEXT NOT NULL PRIMARY KEY,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bars JSONB NOT NULL DEFAULT '[]'::jsonb
);
