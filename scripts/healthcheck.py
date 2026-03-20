#!/usr/bin/env python3
"""Exit 0 if DB is reachable, 1 otherwise. Used by Docker HEALTHCHECK and orchestrators."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from src.db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    sys.exit(main())
