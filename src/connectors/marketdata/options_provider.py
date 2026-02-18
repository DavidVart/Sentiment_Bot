"""Options chain snapshots: Polygon.io (historical EOD) and Tradier sandbox (real-time with Greeks)."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.connectors.marketdata.greeks import DEFAULT_R, _years_to_expiry, compute_greeks
from src.utils.http_utils import with_retry
from src.utils.logging_utils import get_logger
from src.utils.rate_limit import RateLimiter
from src.utils.schemas import OptionsSnapshot

logger = get_logger(__name__)

POLYGON_BASE = "https://api.polygon.io"
TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"
CALLS_PER_MINUTE = 5


def _load_universe() -> list[str]:
    """Load underlyings from configs/universe.yaml."""
    config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs"
    path = config_dir / "universe.yaml"
    if not path.exists():
        return ["SPY", "QQQ", "AAPL"]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("underlyings", ["SPY", "QQQ", "AAPL"]))


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


# --- Polygon: option chain snapshot (current; paginated) ---


class PolygonOptionsProvider:
    """
    Polygon.io options chain snapshot.
    GET /v3/snapshot/options/{underlying}; limit 250, next_url for pagination.
    Free tier: 5 calls/min. Greeks may be missing on some contracts; compute locally with py_vollib.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = POLYGON_BASE,
        calls_per_minute: int = CALLS_PER_MINUTE,
    ) -> None:
        self.api_key = api_key or os.environ.get("MASSIVE_API", "")
        self.base_url = base_url.rstrip("/")
        self._rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)

    def _get(self, path: str, params: dict[str, Any] | None = None, *, full_url: str | None = None) -> Any:
        self._rate_limiter.wait_if_needed()
        url = f"{self.base_url}{path}" if full_url is None else full_url
        params = dict(params or {})
        if self.api_key and "apiKey" not in params:
            params["apiKey"] = self.api_key

        def _request() -> Any:
            with httpx.Client(timeout=60.0) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                return r.json()

        return with_retry(_request)

    def fetch_chain_snapshot(
        self,
        underlying: str,
        snapshot_date: date | None = None,
        expiration_date: str | None = None,
        limit: int = 250,
    ) -> list[OptionsSnapshot]:
        """
        Fetch full options chain snapshot (paginated). snapshot_date defaults to today;
        Polygon returns current snapshot (no historical by-date on this endpoint).
        """
        snapshot_date = snapshot_date or date.today()
        path = f"/v3/snapshot/options/{underlying}"
        params: dict[str, Any] = {"limit": limit}
        if expiration_date:
            params["expiration_date"] = expiration_date
        all_results: list[OptionsSnapshot] = []
        next_url: str | None = None

        while True:
            if next_url:
                data = self._get("", full_url=next_url)
            else:
                data = self._get(path, params)
            results = data.get("results") if isinstance(data, dict) else []
            if not results:
                break
            for r in results:
                if not isinstance(r, dict):
                    continue
                row = self._normalize_row(underlying, snapshot_date, r)
                if row:
                    all_results.append(row)
            next_url = data.get("next_url")
            if not next_url:
                break
        logger.info("Polygon: fetched %s option rows for %s", len(all_results), underlying)
        return all_results

    def _normalize_row(self, underlying: str, snapshot_date: date, r: dict[str, Any]) -> OptionsSnapshot | None:
        details = r.get("details") or {}
        ticker = details.get("ticker")
        if not ticker:
            return None
        expiry = _parse_date(details.get("expiration_date"))
        strike_raw = details.get("strike_price")
        contract_type = (details.get("contract_type") or "call").lower()
        if expiry is None or strike_raw is None:
            return None
        strike = float(strike_raw)
        flag = "c" if contract_type == "call" else "p"

        day = r.get("day") or {}
        close = day.get("close")
        if close is not None:
            close = float(close)
        last_quote = r.get("last_quote") or r.get("last_trade") or {}
        bid = last_quote.get("bid")
        ask = last_quote.get("ask")
        if bid is not None:
            bid = float(bid)
        if ask is not None:
            ask = float(ask)
        mid = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif close is not None:
            mid = close

        iv_raw = r.get("implied_volatility")
        iv = None
        if iv_raw is not None:
            iv = float(iv_raw)
            if iv > 1.0:
                iv = iv / 100.0  # e.g. 25 -> 0.25 (percentage to decimal)
        greeks_in = r.get("greeks") or {}
        delta = greeks_in.get("delta")
        gamma = greeks_in.get("gamma")
        theta = greeks_in.get("theta")
        vega = greeks_in.get("vega")
        if delta is not None:
            delta = float(delta)
        if gamma is not None:
            gamma = float(gamma)
        if theta is not None:
            theta = float(theta)
        if vega is not None:
            vega = float(vega)

        underlying_asset = r.get("underlying_asset") or {}
        S = underlying_asset.get("price")
        if S is not None:
            S = float(S)
        if (delta is None or gamma is None or theta is None or vega is None) and S is not None and iv is not None:
            t = _years_to_expiry(snapshot_date, expiry)
            computed = compute_greeks(flag=flag, S=S, K=strike, t=t, sigma=iv, r=DEFAULT_R)
            if computed:
                delta = delta if delta is not None else computed.get("delta")
                gamma = gamma if gamma is not None else computed.get("gamma")
                theta = theta if theta is not None else computed.get("theta")
                vega = vega if vega is not None else computed.get("vega")

        oi = r.get("open_interest")
        if oi is not None:
            oi = float(oi)
        vol = day.get("volume")
        if vol is not None:
            vol = float(vol)

        return OptionsSnapshot(
            underlying=underlying,
            snapshot_date=snapshot_date,
            contract_id=ticker,
            expiry=expiry,
            strike=strike,
            option_type=contract_type,
            bid=bid,
            ask=ask,
            mid=mid,
            close=close,
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            volume=vol,
            open_interest=oi,
            source="polygon",
        )


# --- Tradier: options chain with Greeks ---


class TradierOptionsProvider:
    """
    Tradier sandbox/production options chain with Greeks (ORATS).
    GET /markets/options/chains?symbol=&expiration=&greeks=true
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str = TRADIER_SANDBOX,
    ) -> None:
        self.api_token = api_token or os.environ.get("TRADIER_API_TOKEN", "")
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}

        def _request() -> Any:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(url, params=params or {}, headers=headers)
                r.raise_for_status()
                return r.json()

        return with_retry(_request)

    def get_expirations(self, symbol: str) -> list[date]:
        """GET /markets/options/expirations?symbol="""
        data = self._get("/markets/options/expirations", {"symbol": symbol})
        dates = data.get("expirations", {}).get("date")
        if not dates:
            return []
        if isinstance(dates, str):
            dates = [dates]
        out = []
        for d in dates:
            parsed = _parse_date(d)
            if parsed:
                out.append(parsed)
        return sorted(out)

    def fetch_chain(
        self,
        underlying: str,
        expiration: date,
        snapshot_date: date | None = None,
        greeks: bool = True,
    ) -> list[OptionsSnapshot]:
        """GET /markets/options/chains?symbol=&expiration=&greeks=true"""
        snapshot_date = snapshot_date or date.today()
        params: dict[str, Any] = {
            "symbol": underlying,
            "expiration": expiration.isoformat(),
            "greeks": "true" if greeks else "false",
        }
        data = self._get("/markets/options/chains", params)
        options = data.get("options")
        if isinstance(options, dict):
            options = options.get("option") or []
        elif options is None:
            options = data.get("option") or []
        if not isinstance(options, list):
            options = [options] if options else []
        out: list[OptionsSnapshot] = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            row = self._normalize_row(underlying, snapshot_date, expiration, opt)
            if row:
                out.append(row)
        logger.info("Tradier: fetched %s option rows for %s exp %s", len(out), underlying, expiration)
        return out

    def _normalize_row(
        self,
        underlying: str,
        snapshot_date: date,
        expiration: date,
        opt: dict[str, Any],
    ) -> OptionsSnapshot | None:
        symbol = opt.get("symbol")
        if not symbol:
            return None
        strike = opt.get("strike")
        if strike is None:
            return None
        strike = float(strike)
        option_type = (opt.get("option_type") or opt.get("type") or "call").lower()

        bid = opt.get("bid")
        ask = opt.get("ask")
        if bid is not None:
            bid = float(bid)
        if ask is not None:
            ask = float(ask)
        mid = opt.get("mid") or (None if (bid is None or ask is None) else (bid + ask) / 2.0)
        close = opt.get("close")

        g = opt.get("greeks") or {}
        if isinstance(g, dict):
            delta = g.get("delta")
            gamma = g.get("gamma")
            theta = g.get("theta")
            vega = g.get("vega")
        else:
            delta = gamma = theta = vega = None
        iv = g.get("smv_vol") or g.get("mid_iv") or opt.get("iv")
        if iv is not None:
            iv = float(iv)
        if delta is not None:
            delta = float(delta)
        if gamma is not None:
            gamma = float(gamma)
        if theta is not None:
            theta = float(theta)
        if vega is not None:
            vega = float(vega)

        volume = opt.get("volume")
        open_interest = opt.get("open_interest")
        if volume is not None:
            volume = float(volume)
        if open_interest is not None:
            open_interest = float(open_interest)

        return OptionsSnapshot(
            underlying=underlying,
            snapshot_date=snapshot_date,
            contract_id=symbol,
            expiry=expiration,
            strike=strike,
            option_type=option_type,
            bid=bid,
            ask=ask,
            mid=mid,
            close=float(close) if close is not None else None,
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            volume=volume,
            open_interest=open_interest,
            source="tradier",
        )


# --- yfinance: free fallback via Yahoo Finance option chains ---


class YFinanceOptionsProvider:
    """
    Yahoo Finance options chains via the ``yfinance`` library.
    No API key required.  Fetches current chains for every available expiration,
    then filters to the *contract menu* from configs/universe.yaml:
        - expiries closest to 7 / 14 / 30 DTE
        - strikes ATM +/- 1σ and +/- 2σ (5 strikes per expiry)
    yfinance provides IV per contract but not Greeks -- those are computed
    locally via ``compute_greeks()`` (py_vollib Black-Scholes).
    """

    def __init__(
        self,
        expiry_targets_days: list[int] | None = None,
        sigma_levels: list[float] | None = None,
    ) -> None:
        self.expiry_targets = expiry_targets_days or [7, 14, 30]
        self.sigma_levels = sigma_levels or [-2.0, -1.0, 0.0, 1.0, 2.0]

    # ---- public API -------------------------------------------------------

    def fetch_chain_snapshot(
        self,
        underlying: str,
        snapshot_date: date | None = None,
    ) -> list[OptionsSnapshot]:
        """Fetch filtered options chain for *underlying* from Yahoo Finance."""
        import numpy as np
        import yfinance as yf

        snapshot_date = snapshot_date or date.today()
        ticker = yf.Ticker(underlying)

        # Underlying price
        spot = self._spot_price(ticker)
        if spot is None or spot <= 0:
            logger.warning("yfinance: cannot determine spot price for %s", underlying)
            return []

        # Available expirations
        try:
            expirations_raw: tuple[str, ...] = ticker.options
        except Exception as exc:
            logger.warning("yfinance: cannot fetch expirations for %s: %s", underlying, exc)
            return []

        if not expirations_raw:
            logger.warning("yfinance: no expirations returned for %s", underlying)
            return []

        # Pick expirations closest to target DTEs
        avail_dates = [date.fromisoformat(d) for d in expirations_raw]
        selected_exps = self._select_expirations(snapshot_date, avail_dates)
        if not selected_exps:
            return []

        # Estimate annualized vol from recent returns (used for σ-strike filter)
        hist_vol = self._estimate_hist_vol(ticker)

        all_rows: list[OptionsSnapshot] = []
        for exp in selected_exps:
            try:
                chain = ticker.option_chain(exp.isoformat())
            except Exception as exc:
                logger.warning("yfinance: option_chain(%s, %s) failed: %s", underlying, exp, exc)
                continue

            # Determine strike range: ATM ± nσ
            dte_years = max((exp - snapshot_date).days / 365.0, 1e-4)
            sigma_1 = spot * hist_vol * np.sqrt(dte_years)
            strike_targets = [spot + lvl * sigma_1 for lvl in self.sigma_levels]

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                if df is None or df.empty:
                    continue
                flag = "c" if opt_type == "call" else "p"

                selected_strikes = self._select_strikes(df["strike"].values, strike_targets)
                df_filt = df[df["strike"].isin(selected_strikes)]

                for _, row in df_filt.iterrows():
                    snap = self._normalize_row(
                        underlying, snapshot_date, exp, opt_type, flag, spot, row,
                    )
                    if snap is not None:
                        all_rows.append(snap)

        logger.info("yfinance: fetched %s option rows for %s (%s expirations)",
                     len(all_rows), underlying, len(selected_exps))
        return all_rows

    # ---- internal helpers --------------------------------------------------

    @staticmethod
    def _spot_price(ticker: Any) -> float | None:
        """Best-effort spot price from yfinance Ticker."""
        try:
            fi = ticker.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
            if price and float(price) > 0:
                return float(price)
        except Exception:
            pass
        try:
            info = ticker.info or {}
            return float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
        except Exception:
            return None

    def _select_expirations(self, today: date, avail: list[date]) -> list[date]:
        """Pick the expiration closest to each target DTE."""
        selected: list[date] = []
        for target_days in self.expiry_targets:
            ideal = date.fromordinal(today.toordinal() + target_days)
            best = min(avail, key=lambda d: abs((d - ideal).days))
            if best not in selected:
                selected.append(best)
        return sorted(selected)

    @staticmethod
    def _select_strikes(available: Any, targets: list[float]) -> list[float]:
        """For each σ-target strike, pick the closest available strike."""
        import numpy as np
        avail_arr = np.asarray(available, dtype=float)
        chosen: set[float] = set()
        for t in targets:
            idx = int(np.argmin(np.abs(avail_arr - t)))
            chosen.add(float(avail_arr[idx]))
        return sorted(chosen)

    @staticmethod
    def _estimate_hist_vol(ticker: Any, window: int = 20) -> float:
        """Annualized realized vol from last *window* daily closes."""
        import numpy as np
        try:
            hist = ticker.history(period="3mo", interval="1d")
            if hist is None or len(hist) < window + 1:
                return 0.20  # fallback
            closes = hist["Close"].dropna().values[-window - 1:]
            log_rets = np.diff(np.log(closes))
            return float(np.std(log_rets) * np.sqrt(252))
        except Exception:
            return 0.20

    def _normalize_row(
        self,
        underlying: str,
        snapshot_date: date,
        expiry: date,
        option_type: str,
        flag: str,
        spot: float,
        row: Any,
    ) -> OptionsSnapshot | None:
        """Map a single yfinance chain row to OptionsSnapshot."""
        contract_symbol = row.get("contractSymbol") if hasattr(row, "get") else getattr(row, "contractSymbol", None)
        strike = float(row.get("strike") if hasattr(row, "get") else getattr(row, "strike", 0))
        if not contract_symbol or strike <= 0:
            return None

        bid = _safe_float(row, "bid")
        ask = _safe_float(row, "ask")
        last = _safe_float(row, "lastPrice")
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else last
        iv = _safe_float(row, "impliedVolatility")
        volume = _safe_float(row, "volume")
        oi = _safe_float(row, "openInterest")

        # Compute Greeks from IV + spot via Black-Scholes
        delta = gamma = theta = vega = None
        if iv is not None and iv > 0:
            t_years = _years_to_expiry(snapshot_date, expiry)
            computed = compute_greeks(flag=flag, S=spot, K=strike, t=t_years, sigma=iv, r=DEFAULT_R)
            if computed:
                delta = computed.get("delta")
                gamma = computed.get("gamma")
                theta = computed.get("theta")
                vega = computed.get("vega")

        return OptionsSnapshot(
            underlying=underlying,
            snapshot_date=snapshot_date,
            contract_id=str(contract_symbol),
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            bid=bid,
            ask=ask,
            mid=mid,
            close=last,
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            volume=volume,
            open_interest=oi,
            source="yfinance",
        )


def _safe_float(row: Any, field: str) -> float | None:
    """Extract a float from a pandas row / dict, returning None for NaN/missing."""
    import math
    val = row.get(field) if hasattr(row, "get") else getattr(row, field, None)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


# --- factory helpers -------------------------------------------------------


def get_polygon_options_provider(api_key: str | None = None) -> PolygonOptionsProvider:
    return PolygonOptionsProvider(api_key=api_key or os.environ.get("MASSIVE_API"))


def get_tradier_options_provider(api_token: str | None = None, sandbox: bool = True) -> TradierOptionsProvider:
    token = api_token or os.environ.get("TRADIER_API_TOKEN")
    base = TRADIER_SANDBOX if sandbox else "https://api.tradier.com/v1"
    return TradierOptionsProvider(api_token=token, base_url=base)


def get_yfinance_options_provider(
    expiry_targets_days: list[int] | None = None,
) -> YFinanceOptionsProvider:
    return YFinanceOptionsProvider(expiry_targets_days=expiry_targets_days)
