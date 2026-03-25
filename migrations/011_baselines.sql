-- Add baselines data to dashboard_evaluation
ALTER TABLE dashboard_evaluation ADD COLUMN IF NOT EXISTS baselines_json JSONB;
