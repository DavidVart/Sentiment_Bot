-- Task run tracking for remote compute monitoring.
-- Stores lifecycle, progress, and logs for ablation / pipeline / report tasks.

CREATE TABLE IF NOT EXISTS task_runs (
    id            SERIAL PRIMARY KEY,
    task_type     TEXT NOT NULL,                          -- 'ablation', 'pipeline', 'reports', 'snapshot'
    task_label    TEXT,                                   -- human-readable, e.g. "Ablation PPO+SAC 5 seeds 50k"
    status        TEXT NOT NULL DEFAULT 'pending',        -- pending | running | completed | failed | cancelled
    progress_pct  REAL DEFAULT 0.0,                      -- 0.0 – 100.0
    current_step  INTEGER DEFAULT 0,
    total_steps   INTEGER DEFAULT 0,
    detail        TEXT,                                   -- e.g. "Run 7/40: variant C / ppo / seed 2 — step 34k/50k"
    log_tail      TEXT,                                   -- last ~30 lines of stdout/stderr
    error_message TEXT,
    pid           INTEGER,                               -- OS process ID (for cancellation)
    config_json   JSONB,                                 -- launch parameters for reproducibility
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_runs_status ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_task_runs_created ON task_runs(created_at DESC);
