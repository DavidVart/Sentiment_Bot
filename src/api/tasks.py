"""
Task runner: create, launch, monitor, and cancel long-running tasks.

Tasks are stored in the ``task_runs`` DB table and executed as subprocesses.
A background thread captures stdout/stderr and periodically updates ``log_tail``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db import get_connection

logger = logging.getLogger(__name__)

# Project root (two levels up from src/api/)
ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable  # use same Python as the running FastAPI process
LOG_TAIL_LINES = 40


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def create_task(task_type: str, config: dict[str, Any], label: str | None = None) -> int:
    """Insert a new task_runs row and return its id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO task_runs (task_type, task_label, status, config_json, created_at)
                VALUES (%s, %s, 'pending', %s, NOW())
                RETURNING id
                """,
                (task_type, label, json.dumps(config)),
            )
            task_id = cur.fetchone()[0]
        conn.commit()
    return task_id


def update_task(task_id: int, **fields) -> None:
    """Update arbitrary fields on a task_runs row."""
    if not fields:
        return
    set_parts = []
    values: list[Any] = []
    for k, v in fields.items():
        set_parts.append(f"{k} = %s")
        values.append(v)
    values.append(task_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE task_runs SET {', '.join(set_parts)} WHERE id = %s",
                values,
            )
        conn.commit()


def complete_task(task_id: int, status: str = "completed", error_message: str | None = None) -> None:
    update_task(
        task_id,
        status=status,
        progress_pct=100.0 if status == "completed" else None,
        completed_at=datetime.now(timezone.utc),
        error_message=error_message,
    )


def get_task(task_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM task_runs WHERE id = %s", (task_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


def list_tasks(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if status:
                cur.execute(
                    "SELECT * FROM task_runs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM task_runs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]


def has_running_task(task_type: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_type = %s AND status = 'running'",
                (task_type,),
            )
            return cur.fetchone()[0] > 0


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _build_command(task_type: str, config: dict[str, Any], task_id: int) -> list[str]:
    """Build the subprocess command for a given task type."""
    base = [PYTHON, "-u"]  # -u = unbuffered stdout

    if task_type == "ablation":
        cmd = base + [str(ROOT / "scripts" / "run_ablation.py")]
        cmd += ["--task-run-id", str(task_id)]
        if config.get("algorithm"):
            cmd += ["--algorithm", str(config["algorithm"])]
        if config.get("seeds"):
            cmd += ["--seeds", str(config["seeds"])]
        if config.get("timesteps"):
            cmd += ["--timesteps", str(config["timesteps"])]
        if config.get("underlying"):
            cmd += ["--underlying", str(config["underlying"])]
        if config.get("out_json"):
            cmd += ["--out-json", str(config["out_json"])]
        if config.get("out_csv"):
            cmd += ["--out-csv", str(config["out_csv"])]
        if config.get("walk_forward"):
            cmd += ["--walk-forward"]
            if config.get("train_days"):
                cmd += ["--train-days", str(config["train_days"])]
            if config.get("eval_days"):
                cmd += ["--eval-days", str(config["eval_days"])]
        return cmd

    elif task_type == "pipeline":
        cmd = base + [str(ROOT / "scripts" / "run_full_pipeline.py")]
        cmd += ["--task-run-id", str(task_id)]
        if config.get("steps"):
            cmd += ["--steps", str(config["steps"])]
        return cmd

    elif task_type == "reports":
        cmd = base + [str(ROOT / "scripts" / "generate_reports.py")]
        cmd += ["--task-run-id", str(task_id)]
        if config.get("ablation_json"):
            cmd += ["--ablation-json", str(config["ablation_json"])]
        if config.get("output_dir"):
            cmd += ["--output-dir", str(config["output_dir"])]
        return cmd

    elif task_type == "snapshot":
        cmd = base + [str(ROOT / "scripts" / "write_dashboard_snapshot.py")]
        cmd += ["--task-run-id", str(task_id)]
        return cmd

    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")


# ---------------------------------------------------------------------------
# Subprocess launcher + log reader
# ---------------------------------------------------------------------------

def _tail_lines(text: str, n: int = LOG_TAIL_LINES) -> str:
    """Keep last n lines."""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _log_reader(proc: subprocess.Popen, task_id: int) -> None:
    """Background thread: read stdout line-by-line and update log_tail in DB."""
    lines: list[str] = []
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            lines.append(line)
            # Periodically flush to DB (every 5 lines or every 10s)
            if len(lines) % 5 == 0:
                try:
                    update_task(task_id, log_tail=_tail_lines("\n".join(lines)))
                except Exception:
                    pass
    except Exception:
        pass
    # Final flush
    try:
        update_task(task_id, log_tail=_tail_lines("\n".join(lines)))
    except Exception:
        pass

    # Wait for process and set final status
    rc = proc.wait()
    try:
        if rc == 0:
            complete_task(task_id, status="completed")
        elif rc == -signal.SIGTERM or rc == -signal.SIGKILL:
            complete_task(task_id, status="cancelled")
        else:
            last_lines = "\n".join(lines[-10:])
            complete_task(task_id, status="failed", error_message=f"exit code {rc}\n{last_lines}")
    except Exception as exc:
        logger.error("Failed to finalize task %d: %s", task_id, exc)


def launch_task(task_id: int) -> int:
    """Launch a task as a subprocess. Returns the OS PID."""
    task = get_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    if task["status"] != "pending":
        raise ValueError(f"Task {task_id} is already {task['status']}")

    config = task["config_json"] if isinstance(task["config_json"], dict) else json.loads(task["config_json"] or "{}")
    cmd = _build_command(task["task_type"], config, task_id)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
    )

    update_task(
        task_id,
        status="running",
        pid=proc.pid,
        started_at=datetime.now(timezone.utc),
    )

    # Start background log reader
    t = threading.Thread(target=_log_reader, args=(proc, task_id), daemon=True)
    t.start()

    logger.info("Launched task %d (PID %d): %s", task_id, proc.pid, " ".join(cmd))
    return proc.pid


def cancel_task(task_id: int) -> bool:
    """Cancel a running task by sending SIGTERM."""
    task = get_task(task_id)
    if not task or task["status"] != "running":
        return False
    pid = task.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        update_task(task_id, status="cancelled", completed_at=datetime.now(timezone.utc))
        return True
    except ProcessLookupError:
        update_task(task_id, status="failed", error_message="Process not found (already exited?)")
        return False
