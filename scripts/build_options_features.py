"""CLI entrypoint: build options_features from options_snapshots and equity_bars."""

from __future__ import annotations

import argparse
import sys
from datetime import date

sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.features.options_features import run_build_options_features
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build options_features (ATM IV, term slope, skew, realized vol, VIX)"
    )
    parser.add_argument("--underlying", type=str, default=None, help="Single underlying (default: all)")
    parser.add_argument("--date", type=str, default=None, help="Feature date YYYY-MM-DD (default: all dates)")
    parser.add_argument("--schema-version", type=int, default=1, help="Schema version for options_features")
    args = parser.parse_args()
    feature_date = date.fromisoformat(args.date) if args.date else None
    run_build_options_features(
        underlying=args.underlying,
        feature_date=feature_date,
        schema_version=args.schema_version,
    )


if __name__ == "__main__":
    main()
