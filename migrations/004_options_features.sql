-- Options-derived features (ATM IV, term structure, skew, realized vol, VIX)
CREATE TABLE IF NOT EXISTS options_features (
    underlying TEXT NOT NULL,
    feature_date DATE NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (underlying, feature_date)
);

CREATE INDEX IF NOT EXISTS idx_options_features_underlying_date ON options_features (underlying, feature_date);
