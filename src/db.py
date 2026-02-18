"""PostgreSQL connection and migration bootstrap."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def get_connection_string() -> str:
    """Build DATABASE_URL from env or from POSTGRES_* vars."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "sentiment_bot")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@contextmanager
def get_connection() -> Generator[PgConnection, None, None]:
    """Context manager for a single DB connection."""
    conn = psycopg2.connect(get_connection_string())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_migration(conn: PgConnection, migration_sql: str) -> None:
    """Execute a migration script (no versioning in Phase 1)."""
    with conn.cursor() as cur:
        cur.execute(migration_sql)
    logger.info("Migration applied.")


def apply_migrations(conn: PgConnection | None = None) -> None:
    """Run all .sql migration files in migrations/ in order."""
    if conn is None:
        with get_connection() as c:
            apply_migrations(c)
        return
    if not MIGRATIONS_DIR.exists():
        logger.warning("Migrations dir not found: %s", MIGRATIONS_DIR)
        return
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for path in files:
        sql_content = path.read_text()
        run_migration(conn, sql_content)
        logger.info("Applied %s", path.name)
