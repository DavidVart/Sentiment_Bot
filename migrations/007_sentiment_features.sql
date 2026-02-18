-- Sentiment features per (underlying, bar_ts): 15-min bars during market hours
CREATE TABLE IF NOT EXISTS sentiment_features (
    underlying TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    sent_news_asset DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_social_asset DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_macro_topic DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_dispersion DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_momentum DOUBLE PRECISION NOT NULL DEFAULT 0,
    sent_volume INT NOT NULL DEFAULT 0,
    no_news_flag BOOLEAN NOT NULL DEFAULT true,
    PRIMARY KEY (underlying, ts)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_features_ts ON sentiment_features (ts);
CREATE INDEX IF NOT EXISTS idx_sentiment_features_underlying ON sentiment_features (underlying);
