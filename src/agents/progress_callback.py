"""
SB3 callback that writes training progress to the task_runs DB table.

Used by the ablation runner to provide real-time progress updates visible
in the /tasks monitoring UI.
"""

from __future__ import annotations

import time
import logging
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def _get_task_connection():
    """Open a standalone psycopg2 connection for progress updates.

    Uses the same env-var logic as src.db but returns a raw connection
    (not a context manager) so it can be kept alive across many callback
    invocations.
    """
    import os
    import psycopg2

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        user = os.environ.get("POSTGRES_USER", "postgres")
        pw = os.environ.get("POSTGRES_PASSWORD", "")
        db = os.environ.get("POSTGRES_DB", "sentiment_bot")
        dsn = f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    return psycopg2.connect(dsn)


class TrainingProgressCallback(BaseCallback):
    """Reports training progress to the ``task_runs`` table.

    Parameters
    ----------
    task_run_id : int
        Row id in ``task_runs`` to update.
    total_timesteps : int
        Total timesteps for this single training run.
    variant : str
        Ablation variant label (A / B / C / D).
    algorithm : str
        Algorithm name (ppo / sac).
    seed : int
        Random seed for this run.
    run_index : int
        0-based index of this run within the full ablation.
    total_runs : int
        Total number of runs in the ablation (e.g. 40 for 4×2×5).
    update_every : int
        Write to DB every N environment steps (default 1000).
    """

    def __init__(
        self,
        task_run_id: int,
        total_timesteps: int,
        variant: str,
        algorithm: str,
        seed: int,
        run_index: int = 0,
        total_runs: int = 1,
        update_every: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.task_run_id = task_run_id
        self.total_timesteps = max(total_timesteps, 1)
        self.variant = variant
        self.algorithm = algorithm
        self.seed = seed
        self.run_index = run_index
        self.total_runs = max(total_runs, 1)
        self.update_every = update_every
        self._conn: Any = None
        self._last_update = 0.0

    # ------------------------------------------------------------------
    def _ensure_conn(self):
        """Lazily open (or reopen) the DB connection."""
        if self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return  # connection alive
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        try:
            self._conn = _get_task_connection()
            self._conn.autocommit = True
        except Exception as exc:
            logger.debug("progress_callback: DB connect failed: %s", exc)
            self._conn = None

    def _write_progress(self, pct: float, detail: str):
        """Write progress to task_runs. Silently skip on DB errors."""
        self._ensure_conn()
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE task_runs
                    SET progress_pct = %s,
                        current_step = %s,
                        total_steps  = %s,
                        detail       = %s
                    WHERE id = %s
                    """,
                    (pct, self.num_timesteps, self.total_timesteps, detail, self.task_run_id),
                )
        except Exception as exc:
            logger.debug("progress_callback: DB write failed: %s", exc)
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        detail = f"Run {self.run_index + 1}/{self.total_runs}: {self.variant}/{self.algorithm}/seed{self.seed} — starting"
        pct = (self.run_index / self.total_runs) * 100
        self._write_progress(pct, detail)

    def _on_step(self) -> bool:
        now = time.monotonic()
        if (self.num_timesteps % self.update_every == 0) or (now - self._last_update > 10):
            self._last_update = now
            frac = self.num_timesteps / self.total_timesteps
            pct = ((self.run_index + frac) / self.total_runs) * 100
            detail = (
                f"Run {self.run_index + 1}/{self.total_runs}: "
                f"{self.variant}/{self.algorithm}/seed{self.seed} — "
                f"step {self.num_timesteps:,}/{self.total_timesteps:,}"
            )
            self._write_progress(pct, detail)
        return True  # continue training

    def _on_training_end(self) -> None:
        pct = ((self.run_index + 1) / self.total_runs) * 100
        detail = (
            f"Run {self.run_index + 1}/{self.total_runs}: "
            f"{self.variant}/{self.algorithm}/seed{self.seed} — done"
        )
        self._write_progress(pct, detail)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
