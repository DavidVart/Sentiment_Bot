"""CLI entrypoint: build pm_features from pm_prices (derived features for ML)."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__file__.resolve().parent.parent))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pm_features from pm_prices")
    parser.add_argument("--token-id", type=str, default=None, help="Single token_id (default: all)")
    parser.add_argument("--schema-version", type=int, default=1, help="Schema version for pm_features")
    args = parser.parse_args()
    # Defer to feature builder module (implemented in pm-feature-builder todo)
    from src.features.pm_feature_builder import run_build_pm_features

    run_build_pm_features(token_id=args.token_id, schema_version=args.schema_version)


if __name__ == "__main__":
    main()
