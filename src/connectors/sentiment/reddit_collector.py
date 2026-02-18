"""Reddit collector using PRAW; implements shared Collector interface."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

from src.utils.logging_utils import get_logger
from src.utils.schemas import Document

logger = get_logger(__name__)

DEFAULT_SUBREDDITS = ["wallstreetbets", "options", "investing", "stocks"]
KNOWN_TICKERS = {"SPY", "QQQ", "AAPL", "S&P", "NASDAQ"}


def _extract_tickers(text: str) -> list[str]:
    """Extract tickers via cashtag ($AAPL) and known symbol matching; dedupe, sorted."""
    if not text:
        return []
    upper = text.upper()
    found: set[str] = set()
    for m in re.finditer(r"\$([A-Z]{1,5})\b", upper):
        found.add(m.group(1))
    for sym in KNOWN_TICKERS:
        if sym in upper or sym.replace("&", "&") in upper:
            if sym == "S&P":
                found.add("SPY")
            else:
                found.add(sym)
    return sorted(found)


def _reddit_timestamp_to_datetime(created_utc: float) -> datetime:
    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
    return dt


def _safe_author(obj: Any) -> str:
    if obj is None:
        return ""
    if hasattr(obj, "name"):
        return str(obj.name) or ""
    return str(obj)[:100] if obj else ""


class RedditCollector:
    """Collect posts and top-level comments from configured subreddits via PRAW."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        subreddits: list[str] | None = None,
        limit_per_sub: int = 100,
    ):
        self.client_id = (client_id or os.environ.get("REDDIT_CLIENT_ID") or "").strip()
        self.client_secret = (client_secret or os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
        self.user_agent = (user_agent or os.environ.get("REDDIT_USER_AGENT") or "").strip()
        if not self.client_id or not self.client_secret or not self.user_agent:
            raise ValueError(
                "Reddit API required: set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT"
            )
        self.subreddits = subreddits or DEFAULT_SUBREDDITS
        self.limit_per_sub = max(1, min(limit_per_sub, 500))
        self._reddit: Any = None

    def _get_reddit(self) -> Any:
        if self._reddit is None:
            try:
                import praw
            except ImportError as e:
                raise ImportError("praw is required for RedditCollector; pip install praw") from e
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
        return self._reddit

    def fetch(self, since_ts: datetime) -> list[Document]:
        """Fetch posts and top-level comments from subreddits with created_utc >= since_ts."""
        since_ts = since_ts.astimezone(timezone.utc)
        since_utc = since_ts.timestamp()
        seen_ids: set[str] = set()
        documents: list[Document] = []

        reddit = self._get_reddit()
        for sub_name in self.subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for submission in sub.new(limit=self.limit_per_sub):
                    if submission.created_utc < since_utc:
                        break
                    doc = self._submission_to_document(submission, sub_name)
                    if doc and doc.id not in seen_ids:
                        seen_ids.add(doc.id)
                        documents.append(doc)
                    # Top-level comments
                    try:
                        submission.comments.replace_more(limit=0)
                    except Exception as e:
                        logger.debug("replace_more failed for %s: %s", submission.id, e)
                    for comment in submission.comments:
                        if getattr(comment, "parent_id", "").startswith("t3_"):
                            c_doc = self._comment_to_document(comment, sub_name, submission.id)
                            if c_doc and c_doc.id not in seen_ids:
                                seen_ids.add(c_doc.id)
                                documents.append(c_doc)
            except Exception as e:
                logger.warning("Reddit subreddit %s failed: %s", sub_name, e)
                continue

        logger.info("Fetched %s Reddit documents (since %s)", len(documents), since_ts.isoformat())
        return documents

    def _submission_to_document(self, submission: Any, subreddit: str) -> Document | None:
        title = (getattr(submission, "title", None) or "").strip()
        if not title and not (getattr(submission, "selftext", None) or "").strip():
            return None
        selftext = (getattr(submission, "selftext", None) or "").strip()
        text = f"{title}. {selftext}".strip() if selftext else title
        url = (getattr(submission, "url", None) or "").strip() or f"https://reddit.com{submission.permalink}"
        ts = _reddit_timestamp_to_datetime(submission.created_utc)
        author = _safe_author(getattr(submission, "author", None))
        score = int(getattr(submission, "score", 0) or 0)
        num_comments = int(getattr(submission, "num_comments", 0) or 0)
        engagement = float(score + num_comments)
        doc_id = f"reddit_post_{submission.id}"
        tickers = _extract_tickers(text)
        return Document(
            id=doc_id,
            source="reddit",
            ts=ts,
            author=author,
            text=text,
            url=url,
            tickers=tickers,
            engagement=engagement,
            language="en",
        )

    def _comment_to_document(self, comment: Any, subreddit: str, submission_id: str) -> Document | None:
        body = (getattr(comment, "body", None) or "").strip()
        if not body or body.lower() in ("[removed]", "[deleted]"):
            return None
        ts = _reddit_timestamp_to_datetime(comment.created_utc)
        author = _safe_author(getattr(comment, "author", None))
        score = int(getattr(comment, "score", 0) or 0)
        engagement = float(score)
        doc_id = f"reddit_comment_{comment.id}"
        tickers = _extract_tickers(body)
        url = f"https://reddit.com/comments/{submission_id}/_/{comment.id}"
        return Document(
            id=doc_id,
            source="reddit",
            ts=ts,
            author=author,
            text=body,
            url=url,
            tickers=tickers,
            engagement=engagement,
            language="en",
        )
