-- Raw sentiment documents (news/social) for reproducibility and re-scoring
CREATE TABLE IF NOT EXISTS sentiment_docs (
    id TEXT NOT NULL PRIMARY KEY,
    source TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    author TEXT DEFAULT '',
    text TEXT NOT NULL,
    url TEXT DEFAULT '',
    tickers TEXT[] DEFAULT '{}',
    engagement DOUBLE PRECISION,
    language TEXT DEFAULT 'en',
    headline_normalized TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_docs_ts ON sentiment_docs (ts);
CREATE INDEX IF NOT EXISTS idx_sentiment_docs_headline ON sentiment_docs (headline_normalized);
CREATE INDEX IF NOT EXISTS idx_sentiment_docs_source ON sentiment_docs (source);
