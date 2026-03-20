#!/usr/bin/env python3
"""Run the data pipeline (step 1 migrations, then 2,3,4,5,6,7,8,9,10,12 ablation) then write the dashboard snapshot. Used by Cloud Run Job or cron."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_full_pipeline.py"), "--steps", "1,2,3,4,5,6,7,8,9,10,12"],
        cwd=str(ROOT),
    )
    # Always run the snapshot so Performance/Trade Impact get updated when step 12 (or a previous run) produced data.
    # If the pipeline failed, we still write pipeline_log and any existing ablation/snapshot data.
    snapshot_r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "write_dashboard_snapshot.py")],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        return r.returncode
    return snapshot_r.returncode


if __name__ == "__main__":
    sys.exit(main())
