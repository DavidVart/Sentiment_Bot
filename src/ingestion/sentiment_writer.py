"""Write sentiment documents to PostgreSQL with headline deduplication (keep earliest)."""

from __future__ import annotations

from datetime import datetime

from psycopg2.extensions import connection as PgConnection

from src.connectors.sentiment.news_collector import _normalize_headline
from src.utils.logging_utils import get_logger
from src.utils.schemas import Document, ScoredDocument, SentimentFeature

logger = get_logger(__name__)


def _headline_from_text(text: str) -> str:
    """First sentence or full text for headline normalization."""
    if not text:
        return ""
    idx = text.find(". ")
    return text[:idx].strip() if idx > 0 else text.strip()


def write_sentiment_docs(conn: PgConnection, documents: list[Document]) -> int:
    """
    Insert documents into sentiment_docs. Deduplication by normalized headline:
    same headline from multiple outlets → keep the one with earliest ts.
    Returns number of rows written.
    """
    written = 0
    with conn.cursor() as cur:
        for doc in documents:
            headline_norm = _normalize_headline(_headline_from_text(doc.text))
            cur.execute(
                "SELECT id, ts FROM sentiment_docs WHERE headline_normalized = %s",
                (headline_norm,),
            )
            row = cur.fetchone()
            ts_str = doc.ts.isoformat()
            if row:
                existing_id, existing_ts = row
                if existing_ts <= doc.ts:
                    continue  # keep existing (earlier)
                cur.execute("DELETE FROM sentiment_docs WHERE id = %s", (existing_id,))
            cur.execute(
                """
                INSERT INTO sentiment_docs
                (id, source, ts, author, text, url, tickers, engagement, language, headline_normalized)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc.id,
                    doc.source,
                    ts_str,
                    doc.author,
                    doc.text,
                    doc.url,
                    doc.tickers,
                    doc.engagement,
                    doc.language,
                    headline_norm,
                ),
            )
            written += 1
    logger.info("Wrote %s sentiment_docs (from %s candidates)", written, len(documents))
    return written


def read_sentiment_docs(
    conn: PgConnection,
    only_unscored: bool = True,
    limit: int | None = None,
) -> list[Document]:
    """Read documents from sentiment_docs. If only_unscored, exclude ids already in sentiment_scored."""
    with conn.cursor() as cur:
        if only_unscored:
            cur.execute(
                """
                SELECT id, source, ts, author, text, url, tickers, engagement, language
                FROM sentiment_docs d
                WHERE NOT EXISTS (SELECT 1 FROM sentiment_scored s WHERE s.id = d.id)
                ORDER BY d.ts
                """ + (" LIMIT %s" if limit is not None else ""),
                (limit,) if limit is not None else (),
            )
        else:
            cur.execute(
                """
                SELECT id, source, ts, author, text, url, tickers, engagement, language
                FROM sentiment_docs
                ORDER BY ts
                """ + (" LIMIT %s" if limit is not None else ""),
                (limit,) if limit is not None else (),
            )
        rows = cur.fetchall()
    docs: list[Document] = []
    for r in rows:
        ts = r[2]
        if isinstance(ts, datetime):
            pass
        else:
            ts = datetime.fromisoformat(ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
        docs.append(
            Document(
                id=r[0],
                source=r[1],
                ts=ts,
                author=r[3] or "",
                text=r[4] or "",
                url=r[5] or "",
                tickers=list(r[6]) if r[6] else [],
                engagement=float(r[7]) if r[7] is not None else None,
                language=r[8] or "en",
            )
        )
    return docs


def write_sentiment_scored(conn: PgConnection, scored: list[ScoredDocument]) -> int:
    """Insert or replace rows in sentiment_scored. Returns number written."""
    with conn.cursor() as cur:
        for s in scored:
            cur.execute(
                """
                INSERT INTO sentiment_scored
                (id, source, ts, author, text, url, tickers, engagement, language,
                 sentiment_pos, sentiment_neg, sentiment_neu, sentiment_compound, sentiment_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    sentiment_pos = EXCLUDED.sentiment_pos,
                    sentiment_neg = EXCLUDED.sentiment_neg,
                    sentiment_neu = EXCLUDED.sentiment_neu,
                    sentiment_compound = EXCLUDED.sentiment_compound,
                    sentiment_model = EXCLUDED.sentiment_model
                """,
                (
                    s.id,
                    s.source,
                    s.ts.isoformat(),
                    s.author,
                    s.text,
                    s.url,
                    s.tickers,
                    s.engagement,
                    s.language,
                    s.sentiment_pos,
                    s.sentiment_neg,
                    s.sentiment_neu,
                    s.sentiment_compound,
                    s.sentiment_model,
                ),
            )
    logger.info("Wrote %s sentiment_scored", len(scored))
    return len(scored)


def write_sentiment_features(
    conn: PgConnection,
    rows: list[SentimentFeature],
    schema_version: int = 1,
) -> None:
    """Upsert sentiment_features (underlying, ts)."""
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO sentiment_features (
                    underlying, ts, schema_version,
                    sent_news_asset, sent_social_asset, sent_macro_topic,
                    sent_dispersion, sent_momentum, sent_volume, no_news_flag
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (underlying, ts) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    sent_news_asset = EXCLUDED.sent_news_asset,
                    sent_social_asset = EXCLUDED.sent_social_asset,
                    sent_macro_topic = EXCLUDED.sent_macro_topic,
                    sent_dispersion = EXCLUDED.sent_dispersion,
                    sent_momentum = EXCLUDED.sent_momentum,
                    sent_volume = EXCLUDED.sent_volume,
                    no_news_flag = EXCLUDED.no_news_flag
                """,
                (
                    row.underlying,
                    row.ts.isoformat(),
                    schema_version,
                    row.sent_news_asset,
                    row.sent_social_asset,
                    row.sent_macro_topic,
                    row.sent_dispersion,
                    row.sent_momentum,
                    row.sent_volume,
                    row.no_news_flag,
                ),
            )
    logger.info("Wrote %s sentiment_features", len(rows))
