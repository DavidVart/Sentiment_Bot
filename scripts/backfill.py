"""CLI entrypoint: backfill prediction-market events, markets, and price history."""

from __future__ import annotations

import argparse
import sys

# Allow running as script: python -m scripts.backfill
sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.ingestion.backfill_pm import run_backfill
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Polymarket + Kalshi to PostgreSQL")
    parser.add_argument("--no-polymarket", action="store_true", help="Skip Polymarket")
    parser.add_argument("--no-kalshi", action="store_true", help="Skip Kalshi")
    parser.add_argument("--max-events", type=int, default=25, help="Max events per platform (default 25)")
    parser.add_argument("--days", type=int, default=365, help="Days of price history (default 365)")
    args = parser.parse_args()
    run_backfill(
        polymarket=not args.no_polymarket,
        kalshi=not args.no_kalshi,
        max_events_per_platform=args.max_events,
        days_history=args.days,
    )


if __name__ == "__main__":
    main()
