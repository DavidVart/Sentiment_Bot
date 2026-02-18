"""Tests for sentiment news collector, Document schema, deduplication, scoring, and writer."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.connectors.sentiment.news_collector import (
    NewsAPICollector,
    _normalize_headline,
)
from src.connectors.sentiment.reddit_collector import (
    RedditCollector,
    _extract_tickers as reddit_extract_tickers,
)
from src.connectors.sentiment.scoring import (
    FinBertScorer,
    VaderScorer,
    score_documents,
)
from src.ingestion.sentiment_writer import (
    read_sentiment_docs,
    write_sentiment_docs,
    write_sentiment_scored,
)
from src.utils.schemas import Document, ScoredDocument


def test_normalize_headline():
    assert _normalize_headline("  Fed Raises  Rates  ") == "fed raises rates"
    assert _normalize_headline("Same Title") == "same title"
    assert _normalize_headline("") == ""


def test_article_to_document():
    """Document built from NewsAPI-like article has expected fields."""
    with patch.dict("os.environ", {"NEWS_API": "test-key"}, clear=False):
        collector = NewsAPICollector(api_key="test-key")
    article = {
        "title": "Apple Stock Rises",
        "description": "AAPL gains on earnings.",
        "url": "https://example.com/a",
        "source": {"id": "example", "name": "Example News"},
        "publishedAt": "2025-02-15T12:00:00Z",
        "author": "Jane Doe",
    }
    doc = collector._article_to_document(article, query="AAPL")
    assert doc is not None
    assert doc.source == "Example News"
    assert doc.author == "Jane Doe"
    assert "Apple" in doc.text and "AAPL" in doc.text
    assert doc.url == "https://example.com/a"
    assert doc.language == "en"
    assert doc.ts.tzinfo is not None
    assert doc.id
    assert "AAPL" in doc.tickers


def test_article_to_document_skips_removed():
    with patch.dict("os.environ", {"NEWS_API": "test-key"}, clear=False):
        collector = NewsAPICollector(api_key="test-key")
    article = {"title": "[Removed]", "url": "https://x.com", "source": {"name": "X"}, "publishedAt": "2025-02-15T12:00:00Z"}
    assert collector._article_to_document(article, query="") is None


def test_fetch_uses_mocked_request():
    """Fetch returns Documents from mocked /everything response."""
    with patch.dict("os.environ", {"NEWS_API": "test-key"}, clear=False):
        collector = NewsAPICollector(api_key="test-key")
    since = datetime(2025, 2, 1, tzinfo=timezone.utc)
    mock_articles = [
        {
            "title": "Inflation Data Due",
            "description": "CPI report tomorrow.",
            "url": "https://example.com/cpi",
            "source": {"name": "Reuters"},
            "publishedAt": "2025-02-10T09:00:00Z",
            "author": "",
        },
    ]
    with patch.object(collector, "_request_everything", return_value=mock_articles):
        docs = collector.fetch(since)
    assert len(docs) == 1
    assert docs[0].source == "Reuters"
    assert "Inflation" in docs[0].text


def test_write_sentiment_docs_dedup_keeps_earliest():
    """Same headline from two outlets: only earliest ts is kept."""
    mock_cur = MagicMock()
    mock_cur.fetchone.side_effect = [
        None,           # first doc: no existing
        ("old-id", datetime(2025, 2, 10, tzinfo=timezone.utc)),  # second: existing earlier
        ("new-id", datetime(2025, 2, 15, tzinfo=timezone.utc)),  # third: existing later -> replace
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None

    early = datetime(2025, 2, 10, 9, 0, tzinfo=timezone.utc)
    late = datetime(2025, 2, 15, 9, 0, tzinfo=timezone.utc)
    docs = [
        Document(id="a", source="A", ts=early, text="Fed Raises Rates.", url="https://a.com"),
        Document(id="b", source="B", ts=late, text="Fed Raises Rates.", url="https://b.com"),   # same headline, later -> skip
        Document(id="c", source="C", ts=early, text="Fed Raises Rates.", url="https://c.com"),   # same headline, earlier -> delete existing, insert
    ]
    written = write_sentiment_docs(mock_conn, docs)
    assert written == 2  # first insert, third replaces; second skipped
    assert mock_cur.execute.call_count >= 4  # SELECTs + INSERTs + one DELETE


def test_write_sentiment_docs_calls_cursor():
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None

    doc = Document(
        id="x",
        source="X",
        ts=datetime.now(timezone.utc),
        text="Headline.",
        url="https://x.com",
    )
    write_sentiment_docs(mock_conn, [doc])
    assert mock_cur.execute.called


# --- Scoring ---


def test_vader_scorer_returns_compound_in_range():
    mock_analyzer = MagicMock()
    mock_analyzer.return_value.polarity_scores.return_value = {
        "pos": 0.2, "neg": 0.3, "neu": 0.5, "compound": -0.1,
    }
    mock_vader_sentiment = MagicMock()
    mock_vader_sentiment.SentimentIntensityAnalyzer = mock_analyzer
    with patch.dict("sys.modules", {"vaderSentiment": MagicMock(), "vaderSentiment.vaderSentiment": mock_vader_sentiment}):
        scorer = VaderScorer()
    doc = Document(id="1", source="x", ts=datetime.now(timezone.utc), text="Stock market crashes today.")
    out = scorer.score(doc)
    assert isinstance(out, ScoredDocument)
    assert out.sentiment_model == "vader"
    assert -1 <= out.sentiment_compound <= 1
    assert 0 <= out.sentiment_pos <= 1 and 0 <= out.sentiment_neg <= 1 and 0 <= out.sentiment_neu <= 1
    assert abs((out.sentiment_pos + out.sentiment_neg + out.sentiment_neu) - 1.0) < 0.01


def test_vader_scorer_positive_text():
    mock_analyzer = MagicMock()
    mock_analyzer.return_value.polarity_scores.return_value = {
        "pos": 0.5, "neg": 0.1, "neu": 0.4, "compound": 0.8,
    }
    mock_vader_sentiment = MagicMock()
    mock_vader_sentiment.SentimentIntensityAnalyzer = mock_analyzer
    with patch.dict("sys.modules", {"vaderSentiment": MagicMock(), "vaderSentiment.vaderSentiment": mock_vader_sentiment}):
        scorer = VaderScorer()
    doc = Document(id="1", source="x", ts=datetime.now(timezone.utc), text="Great earnings and strong growth!")
    out = scorer.score(doc)
    assert out.sentiment_compound > 0
    assert out.sentiment_pos > out.sentiment_neg


def test_finbert_scorer_mocked_pipeline():
    doc = Document(id="1", source="x", ts=datetime.now(timezone.utc), text="Fed raises rates.")
    pipe_callable = MagicMock(return_value=[
        {"label": "positive", "score": 0.1},
        {"label": "negative", "score": 0.7},
        {"label": "neutral", "score": 0.2},
    ])
    def fake_init(self):
        self._pipe = pipe_callable
        self._model_id = "ProsusAI/finbert"
        self._max_length = 512
    with patch.object(FinBertScorer, "__init__", fake_init):
        scorer = FinBertScorer()
    out = scorer.score(doc)
    assert out.sentiment_model == "finbert"
    assert out.sentiment_pos == 0.1 and out.sentiment_neg == 0.7 and out.sentiment_neu == 0.2
    assert out.sentiment_compound == pytest.approx(0.1 - 0.7, abs=0.01)


def test_score_documents_model_vader():
    mock_analyzer = MagicMock()
    mock_analyzer.return_value.polarity_scores.return_value = {"pos": 0.3, "neg": 0.2, "neu": 0.5, "compound": 0.2}
    mock_vader_sentiment = MagicMock()
    mock_vader_sentiment.SentimentIntensityAnalyzer = mock_analyzer
    with patch.dict("sys.modules", {"vaderSentiment": MagicMock(), "vaderSentiment.vaderSentiment": mock_vader_sentiment}):
        docs = [
            Document(id="a", source="reddit", ts=datetime.now(timezone.utc), text="Markets are up."),
        ]
        result = score_documents(docs, model="vader")
    assert len(result) == 1
    assert result[0].sentiment_model == "vader"


def test_score_documents_model_auto_reddit_uses_vader():
    mock_analyzer = MagicMock()
    mock_analyzer.return_value.polarity_scores.return_value = {"pos": 0.3, "neg": 0.2, "neu": 0.5, "compound": 0.2}
    mock_vader_sentiment = MagicMock()
    mock_vader_sentiment.SentimentIntensityAnalyzer = mock_analyzer
    with patch.dict("sys.modules", {"vaderSentiment": MagicMock(), "vaderSentiment.vaderSentiment": mock_vader_sentiment}):
        docs = [
            Document(id="a", source="reddit", ts=datetime.now(timezone.utc), text="Markets are up."),
        ]
        result = score_documents(docs, model="auto")
    assert len(result) == 1
    assert result[0].sentiment_model == "vader"


def test_score_documents_model_auto_newsapi_uses_finbert_when_available():
    docs = [
        Document(id="a", source="newsapi", ts=datetime.now(timezone.utc), text="Fed raises rates."),
    ]
    pipe_callable = MagicMock(return_value=[
        {"label": "positive", "score": 0.2},
        {"label": "negative", "score": 0.3},
        {"label": "neutral", "score": 0.5},
    ])
    def fake_init(self):
        self._pipe = pipe_callable
        self._model_id = "ProsusAI/finbert"
        self._max_length = 512
    mock_analyzer = MagicMock()
    mock_analyzer.return_value.polarity_scores.return_value = {"pos": 0.33, "neg": 0.33, "neu": 0.34, "compound": 0}
    mock_vader_sentiment = MagicMock()
    mock_vader_sentiment.SentimentIntensityAnalyzer = mock_analyzer
    with patch.dict("sys.modules", {"vaderSentiment": MagicMock(), "vaderSentiment.vaderSentiment": mock_vader_sentiment}):
        with patch.object(FinBertScorer, "__init__", fake_init):
            result = score_documents(docs, model="auto")
    assert len(result) == 1
    assert result[0].sentiment_model == "finbert"


def test_score_documents_unknown_model_raises():
    docs = [Document(id="1", source="x", ts=datetime.now(timezone.utc), text="Hi")]
    with pytest.raises(ValueError, match="Unknown model"):
        score_documents(docs, model="unknown")


def test_write_sentiment_scored_calls_cursor():
    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    scored = [
        ScoredDocument(
            id="x", source="x", ts=datetime.now(timezone.utc), text="Hi", url="",
            sentiment_pos=0.2, sentiment_neg=0.1, sentiment_neu=0.7,
            sentiment_compound=0.1, sentiment_model="vader",
        ),
    ]
    write_sentiment_scored(mock_conn, scored)
    assert mock_cur.execute.called


def test_read_sentiment_docs_returns_documents():
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [
        (
            "id1", "Reuters", datetime(2025, 2, 10, tzinfo=timezone.utc),
            "Jane", "Headline.", "https://a.com", ["AAPL"], None, "en",
        ),
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value.__exit__.return_value = None
    docs = read_sentiment_docs(mock_conn, only_unscored=False)
    assert len(docs) == 1
    assert docs[0].id == "id1"
    assert docs[0].source == "Reuters"
    assert docs[0].text == "Headline."


# --- Reddit collector ---


def test_reddit_extract_tickers():
    assert "AAPL" in reddit_extract_tickers("$AAPL to the moon")
    assert "SPY" in reddit_extract_tickers("SPY calls printing")
    assert set(reddit_extract_tickers("$SPY and QQQ")) >= {"SPY", "QQQ"}
    assert reddit_extract_tickers("") == []


def test_reddit_submission_to_document():
    with patch.dict(
        "os.environ",
        {
            "REDDIT_CLIENT_ID": "id",
            "REDDIT_CLIENT_SECRET": "secret",
            "REDDIT_USER_AGENT": "test",
        },
        clear=False,
    ):
        collector = RedditCollector(
            client_id="id", client_secret="secret", user_agent="test"
        )
    mock_sub = MagicMock()
    mock_sub.id = "abc123"
    mock_sub.title = "SPY puts"
    mock_sub.selftext = "What do you think?"
    mock_sub.url = "https://reddit.com/r/wsb/abc"
    mock_sub.permalink = "/r/wallstreetbets/comments/abc123/"
    mock_sub.created_utc = datetime(2025, 2, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    mock_sub.author = MagicMock(name="user1")
    mock_sub.author.name = "user1"
    mock_sub.score = 100
    mock_sub.num_comments = 50
    doc = collector._submission_to_document(mock_sub, "wallstreetbets")
    assert doc is not None
    assert doc.id == "reddit_post_abc123"
    assert doc.source == "reddit"
    assert doc.engagement == 150.0
    assert "SPY" in doc.tickers
    assert doc.author == "user1"


def test_reddit_comment_to_document():
    with patch.dict(
        "os.environ",
        {
            "REDDIT_CLIENT_ID": "id",
            "REDDIT_CLIENT_SECRET": "secret",
            "REDDIT_USER_AGENT": "test",
        },
        clear=False,
    ):
        collector = RedditCollector(
            client_id="id", client_secret="secret", user_agent="test"
        )
    mock_comment = MagicMock()
    mock_comment.id = "c456"
    mock_comment.body = "$AAPL is going up"
    mock_comment.created_utc = datetime(2025, 2, 15, 12, 5, 0, tzinfo=timezone.utc).timestamp()
    mock_comment.author = MagicMock()
    mock_comment.author.name = "commenter"
    mock_comment.score = 10
    doc = collector._comment_to_document(mock_comment, "stocks", "sub123")
    assert doc is not None
    assert doc.id == "reddit_comment_c456"
    assert doc.source == "reddit"
    assert doc.engagement == 10.0
    assert "AAPL" in doc.tickers
    assert "reddit.com" in doc.url


def test_reddit_comment_skips_removed():
    with patch.dict(
        "os.environ",
        {
            "REDDIT_CLIENT_ID": "id",
            "REDDIT_CLIENT_SECRET": "secret",
            "REDDIT_USER_AGENT": "test",
        },
        clear=False,
    ):
        collector = RedditCollector(
            client_id="id", client_secret="secret", user_agent="test"
        )
    mock_comment = MagicMock()
    mock_comment.id = "x"
    mock_comment.body = "[removed]"
    mock_comment.created_utc = 0
    mock_comment.author = None
    mock_comment.score = 0
    assert collector._comment_to_document(mock_comment, "wsb", "sid") is None


def test_reddit_fetch_mocked():
    with patch.dict(
        "os.environ",
        {
            "REDDIT_CLIENT_ID": "id",
            "REDDIT_CLIENT_SECRET": "secret",
            "REDDIT_USER_AGENT": "test",
        },
        clear=False,
    ):
        collector = RedditCollector(
            client_id="id", client_secret="secret", user_agent="test"
        )
    mock_sub = MagicMock()
    mock_sub.id = "s1"
    mock_sub.title = "Markets"
    mock_sub.selftext = ""
    mock_sub.url = "https://reddit.com"
    mock_sub.permalink = "/r/wsb/s1"
    mock_sub.created_utc = datetime(2025, 2, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp()
    mock_sub.author = MagicMock()
    mock_sub.author.name = "u"
    mock_sub.score = 5
    mock_sub.num_comments = 2
    mock_sub.comments = []  # no comments
    mock_sub.fullname = "t3_s1"

    with patch.object(collector, "_get_reddit") as mock_get:
        mock_reddit = MagicMock()
        mock_subreddit = MagicMock()
        mock_subreddit.new.return_value = [mock_sub]
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_get.return_value = mock_reddit
        since = datetime(2025, 2, 15, 0, 0, 0, tzinfo=timezone.utc)
        docs = collector.fetch(since)
    assert len(docs) >= 1
    assert docs[0].source == "reddit"
    assert docs[0].id == "reddit_post_s1"
