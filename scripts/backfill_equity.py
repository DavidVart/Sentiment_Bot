"""CLI entrypoint: backfill equity daily OHLCV (Polygon + yfinance fallback)."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.ingestion.backfill_equity import run_backfill_equity
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill daily equity bars (Polygon.io primary, yfinance fallback)"
    )
    parser.add_argument("--years", type=int, default=2, help="Years of history (default 2)")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to backfill (default: from configs/universe.yaml)",
    )
    args = parser.parse_args()
    run_backfill_equity(years=args.years, symbols=args.symbols or None)


if __name__ == "__main__":
    main()
