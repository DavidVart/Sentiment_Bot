"""CLI entrypoint: backfill options chain snapshots (Polygon + optional Tradier)."""

from __future__ import annotations

import argparse
import sys
from datetime import date

sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.ingestion.backfill_options import run_backfill_options
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill options chain snapshots (Polygon EOD, optional Tradier with Greeks)"
    )
    parser.add_argument("--symbols", nargs="*", default=None, help="Underlyings (default: configs/universe.yaml)")
    parser.add_argument("--date", type=str, default=None, help="Snapshot date YYYY-MM-DD (default: today)")
    parser.add_argument("--tradier", action="store_true", help="Also fetch Tradier sandbox chains (greeks=true)")
    parser.add_argument("--tradier-expirations", type=int, default=3, help="Max Tradier expirations per symbol")
    args = parser.parse_args()
    snapshot_date = date.fromisoformat(args.date) if args.date else None
    run_backfill_options(
        symbols=args.symbols or None,
        snapshot_date=snapshot_date,
        use_tradier=args.tradier,
        tradier_expirations_limit=args.tradier_expirations,
    )


if __name__ == "__main__":
    main()
