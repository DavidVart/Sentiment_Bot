-- Walk-forward validation results: per-fold and aggregate metrics

-- Per-fold results: one row per (variant, algorithm, seed, fold)
CREATE TABLE IF NOT EXISTS walk_forward_results (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_label TEXT,                          -- e.g. 'wf_2026-03-25'
    variant TEXT NOT NULL,                   -- A, B, C, D
    algorithm TEXT NOT NULL,                 -- ppo, sac
    seed INT NOT NULL,
    fold INT NOT NULL,
    train_start_idx INT,
    train_end_idx INT,
    eval_start_idx INT,
    eval_end_idx INT,
    sharpe DOUBLE PRECISION,
    sortino DOUBLE PRECISION,
    calmar DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    hit_rate DOUBLE PRECISION,
    turnover DOUBLE PRECISION,
    total_pnl DOUBLE PRECISION,
    n_bars INT,
    equity_series JSONB,
    regime_metrics JSONB
);

CREATE INDEX IF NOT EXISTS idx_wf_results_variant_algo ON walk_forward_results(variant, algorithm);
CREATE INDEX IF NOT EXISTS idx_wf_results_run_label ON walk_forward_results(run_label);

-- Aggregated walk-forward summary: one row per (variant, algorithm) per run
CREATE TABLE IF NOT EXISTS walk_forward_aggregated (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_label TEXT,
    variant TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    n_folds INT,
    n_seeds INT,
    sharpe_mean DOUBLE PRECISION,
    sharpe_std DOUBLE PRECISION,
    sortino_mean DOUBLE PRECISION,
    sortino_std DOUBLE PRECISION,
    calmar_mean DOUBLE PRECISION,
    calmar_std DOUBLE PRECISION,
    max_drawdown_mean DOUBLE PRECISION,
    max_drawdown_std DOUBLE PRECISION,
    hit_rate_mean DOUBLE PRECISION,
    hit_rate_std DOUBLE PRECISION,
    turnover_mean DOUBLE PRECISION,
    turnover_std DOUBLE PRECISION,
    config_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_wf_agg_run_label ON walk_forward_aggregated(run_label);

-- Dashboard view: precomputed walk-forward summary for the Performance page
ALTER TABLE dashboard_ablation
    ADD COLUMN IF NOT EXISTS walk_forward_json JSONB;
