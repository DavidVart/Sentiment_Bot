"""News collector using NewsAPI; implements shared Collector interface."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
import yaml

from src.utils.http_utils import with_retry
from src.utils.logging_utils import get_logger
from src.utils.schemas import Document

logger = get_logger(__name__)

# Default queries if config missing
DEFAULT_QUERIES = [
    "SPY", "QQQ", "AAPL", "S&P 500", "Nasdaq",
    "Federal Reserve", "CPI", "inflation", "recession", "unemployment",
]


def _normalize_headline(title: str) -> str:
    """Normalize headline for deduplication (lowercase, collapse spaces, strip)."""
    if not title:
        return ""
    s = title.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _load_queries() -> list[str]:
    """Load search queries from configs/sentiment.yaml (universe + macro)."""
    path = Path(__file__).resolve().parents[3] / "configs" / "sentiment.yaml"
    if not path.exists():
        return DEFAULT_QUERIES
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        queries = []
        for key in ("universe", "macro"):
            queries.extend(data.get("news_queries", {}).get(key, []))
        return queries if queries else DEFAULT_QUERIES
    except Exception as e:
        logger.warning("Could not load sentiment config: %s; using defaults", e)
        return DEFAULT_QUERIES


class Collector(Protocol):
    """Shared interface for sentiment document collectors."""

    def fetch(self, since_ts: datetime) -> list[Document]:
        """Fetch documents published on or after since_ts."""
        ...


class NewsAPICollector:
    """Collect news via NewsAPI everything endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://newsapi.org/v2",
        language: str = "en",
        page_size: int = 100,
    ):
        self.api_key = (api_key or os.environ.get("NEWS_API") or "").strip()
        if not self.api_key:
            raise ValueError("NewsAPI key required: set NEWS_API env var")
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.page_size = min(100, max(1, page_size))
        self._queries = _load_queries()

    def fetch(self, since_ts: datetime) -> list[Document]:
        """Fetch articles for all configured queries from since_ts; returns list of Documents."""
        since_ts = since_ts.astimezone(timezone.utc)
        from_param = since_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        seen_ids: set[str] = set()
        documents: list[Document] = []

        for q in self._queries:
            page = 1
            while True:
                batch = self._request_everything(q=q, from_param=from_param, page=page)
                if not batch:
                    break
                for art in batch:
                    doc = self._article_to_document(art, query=q)
                    if doc and doc.id not in seen_ids:
                        seen_ids.add(doc.id)
                        documents.append(doc)
                if len(batch) < self.page_size:
                    break
                page += 1
                if page > 10:  # cap pages per query
                    break

        logger.info("Fetched %s documents from NewsAPI (since %s)", len(documents), from_param)
        return documents

    def _request_everything(
        self,
        q: str,
        from_param: str,
        page: int = 1,
    ) -> list[dict]:
        """One request to /everything; returns list of article dicts or empty on error."""
        url = f"{self.base_url}/everything"
        params = {
            "q": q,
            "from": from_param,
            "language": self.language,
            "pageSize": self.page_size,
            "page": page,
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
        }

        def _get() -> list[dict]:
            r = httpx.get(url, params=params, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                raise RuntimeError(data.get("message", "NewsAPI error"))
            return data.get("articles") or []

        try:
            return with_retry(_get)
        except Exception as e:
            # Don't log full URL (contains apiKey); 401 usually means invalid/expired key
            msg = str(e).split(" for url ")[0] if " for url " in str(e) else str(e)
            logger.warning("NewsAPI request failed for q=%r page=%s: %s", q, page, msg)
            return []

    def _article_to_document(self, article: dict, query: str) -> Document | None:
        """Map one NewsAPI article to Document; id from url+publishedAt+source."""
        title = (article.get("title") or "").strip()
        if not title or title.lower() in ("removed", "[removed]"):
            return None
        desc = (article.get("description") or "").strip()
        text = f"{title}. {desc}".strip() if desc else title
        url = (article.get("url") or "").strip()
        source_obj = article.get("source") or {}
        source_name = (source_obj.get("name") or source_obj.get("id") or "unknown").strip()
        published = article.get("publishedAt")
        try:
            if published.endswith("Z"):
                ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
            else:
                ts = datetime.fromisoformat(published)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None
        author = (article.get("author") or "").strip() or ""
        raw_id = f"{url}|{published}|{source_name}"
        doc_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]
        tickers = _extract_tickers(title, desc, query)
        return Document(
            id=doc_id,
            source=source_name,
            ts=ts,
            author=author,
            text=text,
            url=url,
            tickers=tickers,
            engagement=None,
            language=self.language,
        )


def _extract_tickers(title: str, description: str, query: str) -> list[str]:
    """Extract ticker symbols from title/description and query; dedupe and return sorted."""
    text = f"{title} {description} {query}".upper()
    # Known symbols from our universe + common cashtag pattern
    known = {"SPY", "QQQ", "AAPL", "S&P", "NASDAQ"}
    found: set[str] = set()
    for sym in known:
        if sym in text or sym.replace("&", "&") in text:
            if sym == "S&P":
                found.add("SPY")  # map to ticker
            else:
                found.add(sym)
    # Cashtags like $AAPL
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text):
        found.add(m.group(1))
    return sorted(found)
