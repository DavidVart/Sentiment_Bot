"""Pydantic schemas for external API payloads and normalized internal models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Sentiment: shared document model for news/social ---


class Document(BaseModel):
    """Normalized document for sentiment (news/social)."""

    id: str
    source: str  # outlet or platform name
    ts: datetime
    author: str = ""
    text: str  # headline + description or post body
    url: str = ""
    tickers: list[str] = Field(default_factory=list)
    engagement: float | None = None  # e.g. Reddit score, comment count
    language: str = "en"


class ScoredDocument(Document):
    """Document with sentiment scores from a given model."""

    sentiment_pos: float = 0.0
    sentiment_neg: float = 0.0
    sentiment_neu: float = 0.0
    sentiment_compound: float = 0.0  # typically in [-1, 1]
    sentiment_model: str = ""


class SentimentFeature(BaseModel):
    """Per (underlying, bar_ts) sentiment features aligned to 15-min market-hours clock."""

    underlying: str
    ts: datetime  # bar end time (UTC or ET per convention)
    sent_news_asset: float = 0.0  # FinBERT compound, recency-weighted mean
    sent_social_asset: float = 0.0  # VADER compound, engagement-weighted
    sent_macro_topic: float = 0.0  # FinBERT compound macro-tagged, source-weighted
    sent_dispersion: float = 0.0  # variance across sources
    sent_momentum: float = 0.0  # delta sentiment over trailing window
    sent_volume: int = 0  # count of docs in window
    no_news_flag: bool = False  # True when no docs in window (all sent_* = 0)
    schema_version: int = 1


class FeatureRow(BaseModel):
    """
    Single training row: (underlying, bar_ts) with all sources aligned to 15-min master clock.
    All features at time t use only data available strictly before t (no lookahead).
    """

    underlying: str
    ts: datetime  # bar end time (master clock)
    schema_version: int = 1
    # Options (forward-fill within day; most recent snapshot before bar)
    atm_iv_7d: float | None = None
    atm_iv_14d: float | None = None
    atm_iv_30d: float | None = None
    iv_term_slope: float | None = None
    iv_skew: float | None = None
    realized_vol_5d: float | None = None
    realized_vol_10d: float | None = None
    realized_vol_20d: float | None = None
    realized_vol_60d: float | None = None
    vix_close: float | None = None
    options_gap_flag: bool = False
    # Sentiment (0.0 + no_news_flag if missing)
    sent_news_asset: float = 0.0
    sent_social_asset: float = 0.0
    sent_macro_topic: float = 0.0
    sent_dispersion: float = 0.0
    sent_momentum: float = 0.0
    sent_volume: int = 0
    no_news_flag: bool = True
    # PM (join via mapping; forward-fill < 2h, flag longer gaps)
    pm_p: float | None = None
    pm_logit_p: float | None = None
    pm_delta_p_1h: float | None = None
    pm_delta_p_1d: float | None = None
    pm_momentum: float | None = None
    pm_vol_of_p: float | None = None
    pm_time_to_event: float | None = None
    pm_surprise_z: float | None = None
    pm_gap_flag: bool = False
    # Equity (returns and realized vol from daily bars, no lookahead)
    equity_return_1d: float | None = None
    equity_realized_vol_20d: float | None = None


# --- Normalized prediction-market (platform-agnostic) ---


class PMEvent(BaseModel):
    """Canonical event registry row."""

    event_id: str
    platform: str  # polymarket | kalshi
    title: str
    category: str = ""
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    status: str = ""
    resolution_ts: datetime | None = None


class PMMarket(BaseModel):
    """Market metadata row."""

    market_id: str
    event_id: str
    platform: str
    slug: str = ""
    outcome_names: list[str] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    active: bool = True
    volume: float | None = None
    liquidity: float | None = None


class PMPrice(BaseModel):
    """Raw price timeseries row."""

    token_id: str
    platform: str
    ts: datetime
    price: float  # probability
    mid: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    source: str = "rest"  # rest | websocket


class EquityBar(BaseModel):
    """Normalized daily OHLCV bar for an equity."""

    symbol: str
    ts: date  # bar date (exchange date)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    source: str = "polygon"  # polygon | yfinance


class OptionsSnapshot(BaseModel):
    """Single option contract snapshot (EOD or real-time)."""

    underlying: str
    snapshot_date: date  # bar/snapshot date
    contract_id: str  # OCC symbol or provider ticker (e.g. O:AAPL251219C00150000)
    expiry: date
    strike: float
    option_type: str  # call | put
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    close: float | None = None  # from daily bar when available
    iv: float | None = None  # implied volatility (decimal, e.g. 0.25)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    source: str = "polygon"  # polygon | tradier


class OptionsFeature(BaseModel):
    """Derived options/vol features per underlying per date."""

    underlying: str
    feature_date: date
    atm_iv_7d: float | None = None
    atm_iv_14d: float | None = None
    atm_iv_30d: float | None = None
    iv_term_slope: float | None = None  # IV_30D - IV_7D
    iv_skew: float | None = None  # OTM put IV - OTM call IV
    realized_vol_5d: float | None = None
    realized_vol_10d: float | None = None
    realized_vol_20d: float | None = None
    realized_vol_60d: float | None = None
    vix_close: float | None = None
    schema_version: int = 1


class PMFeature(BaseModel):
    """Derived features for ML."""

    token_id: str
    ts: datetime
    p: float
    logit_p: float
    delta_p_1h: float | None = None
    delta_p_1d: float | None = None
    momentum: float | None = None
    vol_of_p: float | None = None
    time_to_event: float | None = None  # hours
    surprise_z: float | None = None


# --- Polymarket Gamma API (partial) ---


class PolymarketOutcome(BaseModel):
    """Single outcome in a market."""

    name: str | None = None
    price: float | None = None
    token_id: str | None = None


class PolymarketMarket(BaseModel):
    """Market from Gamma API."""

    id: str | None = None
    question: str | None = None
    condition_id: str | None = None
    slug: str | None = None
    outcomes: str | None = None  # sometimes CSV
    outcome_prices: str | None = None  # sometimes CSV
    tokens: list[dict[str, Any]] = Field(default_factory=list)
    volume: float | None = None
    active: bool = True
    end_date_iso: str | None = None
    start_date_iso: str | None = None
    groupItemTitle: str | None = None
    market_slug: str | None = None
    clobTokenIds: list[str] = Field(default_factory=list, alias="clobTokenIds")

    model_config = {"populate_by_name": True}


class PolymarketEvent(BaseModel):
    """Event from Gamma API."""

    id: str | None = None
    title: str | None = None
    slug: str | None = None
    markets: list[PolymarketMarket] = Field(default_factory=list)
    endDate: str | None = None
    startDate: str | None = None
    groupItemTitle: str | None = None

    model_config = {"populate_by_name": True}


# --- Polymarket CLOB (price/book) ---


class ClobPriceRow(BaseModel):
    """Single price from /prices-history or /price."""

    price: float
    timestamp: int | str | None = None  # unix ms or iso


# --- Kalshi API (partial) ---


class KalshiMarket(BaseModel):
    """Market from Kalshi /markets."""

    ticker: str = ""
    event_ticker: str = ""
    title: str | None = None
    yes_bid: int | None = None  # cents
    yes_ask: int | None = None
    no_bid: int | None = None
    no_ask: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    status: str | None = None
    close_time: str | None = None
    expiration_time: str | None = None


class KalshiEvent(BaseModel):
    """Event from Kalshi /events."""

    id: str | None = None
    event_ticker: str = ""
    title: str | None = None
    status: str | None = None
    markets: list[KalshiMarket] = Field(default_factory=list)
    close_time: str | None = None
    expiration_time: str | None = None
