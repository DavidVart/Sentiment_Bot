"""Sentiment scorers: VADER and FinBERT with shared interface."""

from __future__ import annotations

import multiprocessing
import os
import time
from typing import Any, Protocol

from src.utils.logging_utils import get_logger
from src.utils.schemas import Document, ScoredDocument

logger = get_logger(__name__)

MODEL_VADER = "vader"
MODEL_FINBERT = "finbert"

# Maximum seconds to wait for FinBERT model to load (import + download + init)
FINBERT_LOAD_TIMEOUT = int(os.environ.get("FINBERT_LOAD_TIMEOUT", "120"))


class Scorer(Protocol):
    """Shared interface for sentiment scorers."""

    @property
    def model_name(self) -> str:
        ...

    def score(self, doc: Document) -> ScoredDocument:
        """Return a ScoredDocument with sentiment fields set."""
        ...


class VaderScorer:
    """VADER sentiment: compound in [-1, 1]; pos/neg/neu from polarity_scores."""

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
        except ImportError as e:
            raise ImportError("vaderSentiment is required for VaderScorer; pip install vaderSentiment") from e

    @property
    def model_name(self) -> str:
        return MODEL_VADER

    def score(self, doc: Document) -> ScoredDocument:
        scores = self._analyzer.polarity_scores(doc.text)
        return ScoredDocument(
            **doc.model_dump(),
            sentiment_pos=float(scores["pos"]),
            sentiment_neg=float(scores["neg"]),
            sentiment_neu=float(scores["neu"]),
            sentiment_compound=float(scores["compound"]),
            sentiment_model=self.model_name,
        )


def _load_finbert_in_child(result_queue: multiprocessing.Queue, model_id: str) -> None:
    """Target for subprocess: import transformers, load model, signal success."""
    try:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import pipeline  # noqa: delayed import
        _pipe = pipeline("text-classification", model=model_id, top_k=None)
        result_queue.put("ok")
    except Exception as exc:
        result_queue.put(f"error: {exc}")


class FinBertScorer:
    """FinBERT (ProsusAI/finbert) via transformers pipeline; pos/neg/neu + compound = pos - neg.

    The ``transformers`` import and model load is guarded by a timeout.
    If it takes longer than FINBERT_LOAD_TIMEOUT seconds, __init__ raises
    ImportError so the caller can fall back to VADER.
    """

    def __init__(self, model_id: str = "ProsusAI/finbert", max_length: int = 512, timeout: int | None = None):
        timeout = timeout if timeout is not None else FINBERT_LOAD_TIMEOUT
        self._model_id = model_id
        self._max_length = max_length

        # First, warm-check: try loading in a child process with a timeout.
        # If the child doesn't finish in time (slow transformers import),
        # we abort and raise so the caller falls back to VADER.
        if not self._warm_check(model_id, timeout):
            raise ImportError(
                f"FinBERT loading timed out after {timeout}s "
                "(slow transformers import); falling back to VADER"
            )

        # If warm check passed, load in this process (should be fast now due to caching)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            from transformers import pipeline as tf_pipeline
            self._pipe = tf_pipeline(
                "text-classification",
                model=model_id,
                top_k=None,
            )
        except Exception as e:
            raise ImportError(
                "transformers and a compatible backend (e.g. torch) are required for FinBertScorer"
            ) from e

    @staticmethod
    def _warm_check(model_id: str, timeout: int) -> bool:
        """Try loading FinBERT in a child process; return True if it finishes in time."""
        q: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_load_finbert_in_child, args=(q, model_id), daemon=True,
        )
        t0 = time.monotonic()
        proc.start()
        proc.join(timeout=timeout)
        elapsed = time.monotonic() - t0

        if proc.is_alive():
            logger.warning(
                "FinBERT warm-check still running after %.0fs; killing child", elapsed
            )
            proc.kill()
            proc.join(timeout=5)
            return False

        if proc.exitcode != 0:
            logger.warning("FinBERT warm-check child exited with code %s", proc.exitcode)
            return False

        try:
            result = q.get_nowait()
        except Exception:
            result = "no result"

        if result == "ok":
            logger.info("FinBERT warm-check passed in %.1fs", elapsed)
            return True

        logger.warning("FinBERT warm-check failed: %s", result)
        return False

    @property
    def model_name(self) -> str:
        return MODEL_FINBERT

    def score(self, doc: Document) -> ScoredDocument:
        text = (doc.text or "")[: self._max_length * 2]
        if not text.strip():
            return ScoredDocument(
                **doc.model_dump(),
                sentiment_pos=0.0,
                sentiment_neg=0.0,
                sentiment_neu=1.0,
                sentiment_compound=0.0,
                sentiment_model=self.model_name,
            )
        try:
            out = self._pipe(text, max_length=self._max_length, truncation=True, top_k=None)
        except Exception as e:
            logger.warning("FinBERT scoring failed for doc %s: %s", doc.id, e)
            return ScoredDocument(
                **doc.model_dump(),
                sentiment_pos=0.0,
                sentiment_neg=0.0,
                sentiment_neu=1.0,
                sentiment_compound=0.0,
                sentiment_model=self.model_name,
            )
        pos = neg = neu = 0.0
        if isinstance(out, list) and len(out) > 0:
            items = out[0] if isinstance(out[0], list) else out
            for item in items:
                if not isinstance(item, dict):
                    continue
                label = (item.get("label") or "").lower()
                score_val = float(item.get("score", 0))
                if "pos" in label:
                    pos = score_val
                elif "neg" in label:
                    neg = score_val
                else:
                    neu = score_val
        compound = pos - neg
        return ScoredDocument(
            **doc.model_dump(),
            sentiment_pos=pos,
            sentiment_neg=neg,
            sentiment_neu=neu,
            sentiment_compound=compound,
            sentiment_model=self.model_name,
        )


def score_documents(
    docs: list[Document],
    model: str = "auto",
) -> list[ScoredDocument]:
    """
    Score documents with the given model.
    - model='auto': FinBERT for source='newsapi', VADER for source='reddit'; otherwise VADER.
      If FinBERT fails to load within the timeout, all docs are scored with VADER.
    - model='vader' | 'finbert': use that scorer for all.
    """
    if not docs:
        return []
    if model == "auto":
        vader = VaderScorer()
        finbert: FinBertScorer | None = None
        try:
            t0 = time.monotonic()
            finbert = FinBertScorer()
            logger.info("FinBERT loaded in %.1fs", time.monotonic() - t0)
        except Exception as exc:
            logger.warning("FinBERT unavailable, scoring all docs with VADER: %s", exc)
            finbert = None
        result: list[ScoredDocument] = []
        for doc in docs:
            if (doc.source or "").lower() == "newsapi" and finbert is not None:
                result.append(finbert.score(doc))
            else:
                result.append(vader.score(doc))
        return result
    if model == MODEL_VADER:
        scorer = VaderScorer()
        return [scorer.score(d) for d in docs]
    if model == MODEL_FINBERT:
        scorer_fb = FinBertScorer()
        return [scorer_fb.score(d) for d in docs]
    raise ValueError(f"Unknown model: {model}; use 'auto', 'vader', or 'finbert'")
