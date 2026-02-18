-- Equity daily OHLCV bars (Polygon / yfinance)
CREATE TABLE IF NOT EXISTS equity_bars (
    symbol TEXT NOT NULL,
    ts DATE NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    vwap DOUBLE PRECISION,
    source TEXT DEFAULT 'polygon',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_equity_bars_symbol_ts ON equity_bars (symbol, ts);
