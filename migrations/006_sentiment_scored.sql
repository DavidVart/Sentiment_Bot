-- Scored sentiment documents (model and pos/neg/neu/compound)
CREATE TABLE IF NOT EXISTS sentiment_scored (
    id TEXT NOT NULL PRIMARY KEY,
    source TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    author TEXT DEFAULT '',
    text TEXT NOT NULL,
    url TEXT DEFAULT '',
    tickers TEXT[] DEFAULT '{}',
    engagement DOUBLE PRECISION,
    language TEXT DEFAULT 'en',
    sentiment_pos DOUBLE PRECISION NOT NULL DEFAULT 0,
    sentiment_neg DOUBLE PRECISION NOT NULL DEFAULT 0,
    sentiment_neu DOUBLE PRECISION NOT NULL DEFAULT 0,
    sentiment_compound DOUBLE PRECISION NOT NULL DEFAULT 0,
    sentiment_model TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_scored_ts ON sentiment_scored (ts);
CREATE INDEX IF NOT EXISTS idx_sentiment_scored_model ON sentiment_scored (sentiment_model);
CREATE INDEX IF NOT EXISTS idx_sentiment_scored_source ON sentiment_scored (source);
