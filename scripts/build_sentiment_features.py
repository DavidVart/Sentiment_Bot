"""CLI entrypoint: build sentiment_features from sentiment_scored (15-min bars, market hours)."""

from __future__ import annotations

import argparse
import sys
from datetime import date

sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.features.sentiment_features import run_build_sentiment_features
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sentiment_features from sentiment_scored (15-min bars 9:30–16:00 ET)"
    )
    parser.add_argument("--underlying", type=str, default=None, help="Single underlying (default: all from universe)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD (default: from data)")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (default: from data)")
    parser.add_argument("--schema-version", type=int, default=1, help="Schema version for sentiment_features")
    parser.add_argument("--migrate", action="store_true", help="Apply migrations before building")
    args = parser.parse_args()
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    if args.migrate:
        from src.db import apply_migrations
        apply_migrations()
    run_build_sentiment_features(
        underlying=args.underlying,
        start_date=start_date,
        end_date=end_date,
        schema_version=args.schema_version,
    )


if __name__ == "__main__":
    main()
