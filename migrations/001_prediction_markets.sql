-- Prediction market tables (platform-agnostic: Polymarket + Kalshi)
-- Schema version: 1

CREATE TABLE IF NOT EXISTS pm_events (
    event_id TEXT NOT NULL PRIMARY KEY,
    platform TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT DEFAULT '',
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    status TEXT DEFAULT '',
    resolution_ts TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pm_markets (
    market_id TEXT NOT NULL PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES pm_events(event_id),
    platform TEXT NOT NULL,
    slug TEXT DEFAULT '',
    outcome_names TEXT[] DEFAULT '{}',
    token_ids TEXT[] DEFAULT '{}',
    active BOOLEAN DEFAULT TRUE,
    volume DOUBLE PRECISION,
    liquidity DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pm_prices (
    id BIGSERIAL PRIMARY KEY,
    token_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    mid DOUBLE PRECISION,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    source TEXT DEFAULT 'rest',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pm_prices_token_ts ON pm_prices (token_id, ts);
CREATE INDEX IF NOT EXISTS idx_pm_prices_platform_ts ON pm_prices (platform, ts);

CREATE TABLE IF NOT EXISTS pm_features (
    id BIGSERIAL PRIMARY KEY,
    token_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    p DOUBLE PRECISION NOT NULL,
    logit_p DOUBLE PRECISION NOT NULL,
    delta_p_1h DOUBLE PRECISION,
    delta_p_1d DOUBLE PRECISION,
    momentum DOUBLE PRECISION,
    vol_of_p DOUBLE PRECISION,
    time_to_event DOUBLE PRECISION,
    surprise_z DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (token_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_pm_features_token_ts ON pm_features (token_id, ts);
