-- Master-aligned feature matrix: one row per (underlying, ts) on 15-min market-hours clock
-- Schema version must be bumped for any column change; requires full recomputation
CREATE TABLE IF NOT EXISTS feature_bars (
    underlying TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    -- Options (forward-fill within day; gap if no snapshot before bar)
    atm_iv_7d DOUBLE PRECISION,
    atm_iv_14d DOUBLE PRECISION,
    atm_iv_30d DOUBLE PRECISION,
    iv_term_slope DOUBLE PRECISION,
    iv_skew DOUBLE PRECISION,
    realized_vol_5d DOUBLE PRECISION,
    realized_vol_10d DOUBLE PRECISION,
    realized_vol_20d DOUBLE PRECISION,
    realized_vol_60d DOUBLE PRECISION,
    vix_close DOUBLE PRECISION,
    options_gap_flag BOOLEAN NOT NULL DEFAULT true,
    -- Sentiment (0.0 + no_news_flag if missing)
    sent_news_asset DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_social_asset DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_macro_topic DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_dispersion DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_momentum DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_volume INT NOT NULL DEFAULT 0,
    no_news_flag BOOLEAN NOT NULL DEFAULT true,
    -- PM (forward-fill < 2h; gap if longer)
    pm_p DOUBLE PRECISION,
    pm_logit_p DOUBLE PRECISION,
    pm_delta_p_1h DOUBLE PRECISION,
    pm_delta_p_1d DOUBLE PRECISION,
    pm_momentum DOUBLE PRECISION,
    pm_vol_of_p DOUBLE PRECISION,
    pm_time_to_event DOUBLE PRECISION,
    pm_surprise_z DOUBLE PRECISION,
    pm_gap_flag BOOLEAN NOT NULL DEFAULT true,
    -- Equity (daily returns/vol, no lookahead)
    equity_return_1d DOUBLE PRECISION,
    equity_realized_vol_20d DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (underlying, ts)
);

CREATE INDEX IF NOT EXISTS idx_feature_bars_ts ON feature_bars (ts);
CREATE INDEX IF NOT EXISTS idx_feature_bars_underlying ON feature_bars (underlying);
CREATE INDEX IF NOT EXISTS idx_feature_bars_schema_version ON feature_bars (schema_version);
