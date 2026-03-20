#!/bin/bash
# Run remaining capstone steps (3-5) sequentially with fixed BS pricing environment.
# Usage: nohup bash scripts/run_remaining_steps.sh > remaining_steps.log 2>&1 &
# Then close laptop. Check progress with: tail -f remaining_steps.log

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export $(grep -v '^#' .env | xargs 2>/dev/null) 2>/dev/null || true

echo ""
echo "=========================================="
echo "$(date) | FRESH RUN with fixed BS pricing environment"
echo "=========================================="

echo "=========================================="
echo "$(date) | Step 3: Ablation study starting (PPO+SAC, 5 seeds, 50k steps)..."
echo "=========================================="

python -u scripts/run_ablation.py \
    --algorithm both --seeds 5 --timesteps 50000 \
    --out-json ablation_results.json --out-csv ablation_results.csv

echo ""
echo "=========================================="
echo "$(date) | Step 3 DONE. Checking results..."
echo "=========================================="
ls -la ablation_results.json ablation_results.csv

# Verify non-zero results
python -c "
import csv
with open('ablation_results.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
non_zero = sum(1 for r in rows if float(r['sharpe']) != 0.0)
print(f'  {non_zero}/{len(rows)} runs have non-zero Sharpe')
"

echo ""
echo "=========================================="
echo "$(date) | Step 4: Generating reports..."
echo "=========================================="
python -u scripts/generate_reports.py \
    --ablation-json ablation_results.json \
    --output-dir reports \
    --feature-bars-db 1

echo ""
echo "=========================================="
echo "$(date) | Step 4 DONE. Reports generated:"
echo "=========================================="
ls -la reports/
echo ""

echo "=========================================="
echo "$(date) | Step 5: Writing dashboard snapshots..."
echo "=========================================="
python -u scripts/write_dashboard_snapshot.py

echo ""
echo "=========================================="
echo "$(date) | ALL STEPS COMPLETE!"
echo "=========================================="
echo "Summary:"
echo "  - ablation_results.json: $(wc -c < ablation_results.json) bytes"
echo "  - ablation_results.csv:  $(wc -c < ablation_results.csv) bytes"
echo "  - reports/ contents:     $(ls reports/ 2>/dev/null | wc -l) files"
echo ""

# macOS notification
osascript -e 'display notification "All capstone steps (3-5) are DONE! Check remaining_steps.log" with title "Sentiment Bot ✅" sound name "Hero"' 2>/dev/null || true
