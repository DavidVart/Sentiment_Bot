"""CLI entrypoint: incremental poll of prediction-market prices -> pm_prices."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.ingestion.incremental_pm import run_incremental
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental update: poll latest PM prices and write to DB")
    parser.add_argument("--polymarket-tokens", nargs="*", default=None, help="Polymarket token IDs (default: from mapping.yaml)")
    parser.add_argument("--kalshi-tickers", nargs="*", default=None, help="Kalshi market tickers (default: from mapping.yaml)")
    args = parser.parse_args()
    run_incremental(
        polymarket_token_ids=args.polymarket_tokens,
        kalshi_tickers=args.kalshi_tickers,
    )


if __name__ == "__main__":
    main()
