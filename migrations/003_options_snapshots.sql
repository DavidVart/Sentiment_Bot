-- Options chain snapshots (Polygon EOD / Tradier real-time)
CREATE TABLE IF NOT EXISTS options_snapshots (
    underlying TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    contract_id TEXT NOT NULL,
    expiry DATE NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    option_type TEXT NOT NULL,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    mid DOUBLE PRECISION,
    close DOUBLE PRECISION,
    iv DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    source TEXT DEFAULT 'polygon',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (underlying, snapshot_date, contract_id)
);

CREATE INDEX IF NOT EXISTS idx_options_snapshots_underlying_date ON options_snapshots (underlying, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_options_snapshots_expiry ON options_snapshots (underlying, expiry);
