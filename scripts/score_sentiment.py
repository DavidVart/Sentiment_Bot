"""Score sentiment_docs and write results to sentiment_scored."""

from __future__ import annotations

import argparse

from src.connectors.sentiment import score_documents
from src.db import apply_migrations, get_connection
from src.ingestion.sentiment_writer import read_sentiment_docs, write_sentiment_scored
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sentiment_docs and write to sentiment_scored")
    parser.add_argument(
        "--model",
        choices=("auto", "vader", "finbert"),
        default="auto",
        help="Scoring model: auto (FinBERT for newsapi, VADER else), vader, or finbert",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all docs in sentiment_docs (default: only those not yet in sentiment_scored)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of docs to score")
    parser.add_argument("--migrate", action="store_true", help="Apply migrations before scoring")
    args = parser.parse_args()

    if args.migrate:
        apply_migrations()

    with get_connection() as conn:
        docs = read_sentiment_docs(conn, only_unscored=not args.all, limit=args.limit)
    if not docs:
        logger.info("No documents to score")
        return
    scored = score_documents(docs, model=args.model)
    with get_connection() as conn:
        written = write_sentiment_scored(conn, scored)
    logger.info("Scored and wrote %s documents", written)


if __name__ == "__main__":
    main()
