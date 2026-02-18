"""Backfill sentiment_docs from NewsAPI and/or Reddit (since a given timestamp or last N days)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.connectors.sentiment import NewsAPICollector, RedditCollector
from src.db import apply_migrations, get_connection
from src.ingestion.sentiment_writer import write_sentiment_docs
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill sentiment_docs from NewsAPI and/or Reddit")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Fetch from the last N days (default 7)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO datetime for oldest item (overrides --days)",
    )
    parser.add_argument(
        "--source",
        choices=("news", "reddit", "all"),
        default="all",
        help="Source to backfill: news (NewsAPI), reddit (PRAW), or all (default)",
    )
    parser.add_argument("--migrate", action="store_true", help="Apply migrations before backfill")
    args = parser.parse_args()

    if args.since:
        try:
            since_ts = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
            if since_ts.tzinfo is None:
                since_ts = since_ts.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error("Invalid --since value; use ISO format")
            raise SystemExit(1)
    else:
        since_ts = datetime.now(timezone.utc) - timedelta(days=args.days)

    if args.migrate:
        apply_migrations()

    documents: list = []
    if args.source in ("news", "all"):
        try:
            collector = NewsAPICollector()
            documents.extend(collector.fetch(since_ts))
        except ValueError as e:
            logger.warning("Skipping NewsAPI: %s", e)
    if args.source in ("reddit", "all"):
        try:
            reddit = RedditCollector()
            documents.extend(reddit.fetch(since_ts))
        except ValueError as e:
            logger.warning("Skipping Reddit: %s", e)

    if documents:
        with get_connection() as conn:
            written = write_sentiment_docs(conn, documents)
        logger.info("Backfill complete: %s documents written", written)
    else:
        logger.info("No documents to write")


if __name__ == "__main__":
    main()
