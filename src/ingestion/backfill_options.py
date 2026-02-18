"""Backfill options chain snapshots: Polygon (EOD snapshot), yfinance subprocess fallback, optional Tradier."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.connectors.marketdata.options_provider import (
    PolygonOptionsProvider,
    TradierOptionsProvider,
    get_polygon_options_provider,
    get_tradier_options_provider,
)
from src.db import apply_migrations, get_connection
from src.ingestion.pm_writer import write_options_snapshots
from src.utils.logging_utils import get_logger
from src.utils.schemas import OptionsSnapshot

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
PYTHON_BIN = str(Path(sys.executable))


def load_data_sources() -> dict[str, Any]:
    path = CONFIG_DIR / "data_sources.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_universe() -> list[str]:
    path = CONFIG_DIR / "universe.yaml"
    if not path.exists():
        return ["SPY", "QQQ", "AAPL"]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("underlyings", ["SPY", "QQQ", "AAPL"]))


def _try_polygon(
    polygon: PolygonOptionsProvider,
    underlying: str,
    snapshot_date: date,
) -> tuple[list[OptionsSnapshot], bool]:
    """Attempt Polygon fetch.  Returns (rows, is_auth_error)."""
    try:
        rows = polygon.fetch_chain_snapshot(underlying, snapshot_date=snapshot_date)
        return rows, False
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            logger.info(
                "Polygon returned %s for %s; will fall back to yfinance",
                exc.response.status_code,
                underlying,
            )
            return [], True
        logger.warning("Polygon snapshot %s failed: %s", underlying, exc)
        return [], False
    except Exception as exc:
        logger.warning("Polygon snapshot %s failed: %s", underlying, exc)
        return [], False


def _yfinance_subprocess(
    underlying: str,
    snapshot_date: date,
    timeout: int = 120,
) -> list[OptionsSnapshot]:
    """
    Run yfinance fetch in a separate process to isolate bus errors / segfaults.

    Calls scripts/fetch_yfinance_options.py via subprocess.  If the child
    process crashes (bus error, segfault, timeout), this function returns an
    empty list instead of crashing the pipeline.
    """
    script = str(SCRIPTS_DIR / "fetch_yfinance_options.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        output_path = tmp.name

    cmd = [
        PYTHON_BIN, script,
        "--symbol", underlying,
        "--snapshot-date", snapshot_date.isoformat(),
        "--output", output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Log subprocess stderr (contains progress info)
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.info("yfinance[%s]: %s", underlying, line)

        if result.returncode != 0:
            logger.warning(
                "yfinance subprocess for %s exited with code %s (signal %s)",
                underlying,
                result.returncode,
                -result.returncode if result.returncode < 0 else "n/a",
            )
            return []

        # Read JSON output
        out_path = Path(output_path)
        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning("yfinance subprocess for %s produced no output", underlying)
            return []

        raw = json.loads(out_path.read_text())
        rows = []
        for d in raw:
            d["snapshot_date"] = date.fromisoformat(d["snapshot_date"])
            d["expiry"] = date.fromisoformat(d["expiry"])
            rows.append(OptionsSnapshot(**d))
        return rows

    except subprocess.TimeoutExpired:
        logger.warning("yfinance subprocess for %s timed out after %ss", underlying, timeout)
        return []
    except Exception as exc:
        logger.warning("yfinance subprocess for %s failed: %s", underlying, exc)
        return []
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass


def run_backfill_options(
    symbols: list[str] | None = None,
    snapshot_date: date | None = None,
    use_tradier: bool = False,
    tradier_expirations_limit: int = 3,
) -> None:
    """
    Backfill options_snapshots.

    Strategy: try Polygon first for each underlying.  If Polygon returns 403
    (free-tier restriction) fall back to yfinance via a subprocess for that
    underlying.  The subprocess isolates yfinance so that bus errors / segfaults
    on Apple Silicon don't crash the entire pipeline.
    Optionally also fetch Tradier chains for nearest expirations.
    """
    apply_migrations()
    symbols = symbols or load_universe()
    snapshot_date = snapshot_date or date.today()
    all_rows: list[OptionsSnapshot] = []

    polygon = get_polygon_options_provider()
    polygon_blocked = False

    for underlying in symbols:
        if not polygon_blocked:
            rows, auth_err = _try_polygon(polygon, underlying, snapshot_date)
            if auth_err:
                polygon_blocked = True
            else:
                all_rows.extend(rows)

        if polygon_blocked:
            logger.info("Using yfinance subprocess for %s", underlying)
            rows = _yfinance_subprocess(underlying, snapshot_date)
            all_rows.extend(rows)

    if use_tradier:
        try:
            tradier = get_tradier_options_provider(sandbox=True)
            for underlying in symbols:
                try:
                    expirations = tradier.get_expirations(underlying)
                    for exp in expirations[:tradier_expirations_limit]:
                        rows = tradier.fetch_chain(underlying, exp, snapshot_date=snapshot_date, greeks=True)
                        all_rows.extend(rows)
                except Exception as e:
                    logger.warning("Tradier chain %s failed: %s", underlying, e)
        except Exception as e:
            logger.warning("Tradier provider failed: %s", e)

    if all_rows:
        with get_connection() as conn:
            write_options_snapshots(conn, all_rows)
        logger.info("Options backfill complete: %s rows (source mix: %s)",
                     len(all_rows), _source_summary(all_rows))
    else:
        logger.warning("Options backfill: no rows (check MASSIVE_API or yfinance availability)")


def _source_summary(rows: list[OptionsSnapshot]) -> str:
    """Summarize source counts, e.g. 'polygon=120, yfinance=30'."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.source] = counts.get(r.source, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
