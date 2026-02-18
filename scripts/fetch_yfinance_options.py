#!/usr/bin/env python3
"""
Subprocess helper: fetch options chains via yfinance for a single symbol.

Called by backfill_options.py to isolate yfinance in a separate process,
avoiding bus errors / segfaults on Apple Silicon from crashing the pipeline.

Usage:
    python scripts/fetch_yfinance_options.py \
        --symbol SPY --snapshot-date 2026-02-17 --output /tmp/spy_options.json

Writes a JSON array of OptionsSnapshot dicts to --output.
Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def _safe_float(row, field: str):
    val = row.get(field) if hasattr(row, "get") else getattr(row, field, None)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def fetch_options(symbol: str, snapshot_date: date, expiry_targets: list[int], sigma_levels: list[float]) -> list[dict]:
    import numpy as np
    import yfinance as yf

    from src.connectors.marketdata.greeks import DEFAULT_R, _years_to_expiry, compute_greeks

    ticker = yf.Ticker(symbol)

    # Spot price
    spot = None
    try:
        fi = ticker.fast_info
        spot = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
        if spot:
            spot = float(spot)
    except Exception:
        pass
    if not spot or spot <= 0:
        try:
            info = ticker.info or {}
            spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
        except Exception:
            pass
    if not spot or spot <= 0:
        print(f"Cannot determine spot price for {symbol}", file=sys.stderr)
        return []

    # Expirations
    try:
        expirations_raw = ticker.options
    except Exception as exc:
        print(f"Cannot fetch expirations for {symbol}: {exc}", file=sys.stderr)
        return []

    if not expirations_raw:
        print(f"No expirations returned for {symbol}", file=sys.stderr)
        return []

    avail_dates = [date.fromisoformat(d) for d in expirations_raw]

    # Select expirations closest to target DTEs
    selected_exps = []
    for target_days in expiry_targets:
        ideal = date.fromordinal(snapshot_date.toordinal() + target_days)
        best = min(avail_dates, key=lambda d: abs((d - ideal).days))
        if best not in selected_exps:
            selected_exps.append(best)
    selected_exps.sort()

    if not selected_exps:
        return []

    # Historical vol for strike filtering
    hist_vol = 0.20
    try:
        hist = ticker.history(period="3mo", interval="1d")
        if hist is not None and len(hist) >= 21:
            closes = hist["Close"].dropna().values[-21:]
            log_rets = np.diff(np.log(closes))
            hist_vol = float(np.std(log_rets) * np.sqrt(252))
    except Exception:
        pass

    all_rows = []
    for exp in selected_exps:
        try:
            chain = ticker.option_chain(exp.isoformat())
        except Exception as exc:
            print(f"option_chain({symbol}, {exp}) failed: {exc}", file=sys.stderr)
            continue

        dte_years = max((exp - snapshot_date).days / 365.0, 1e-4)
        sigma_1 = spot * hist_vol * np.sqrt(dte_years)
        strike_targets = [spot + lvl * sigma_1 for lvl in sigma_levels]

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            if df is None or df.empty:
                continue
            flag = "c" if opt_type == "call" else "p"

            avail_arr = np.asarray(df["strike"].values, dtype=float)
            chosen_strikes = set()
            for t in strike_targets:
                idx = int(np.argmin(np.abs(avail_arr - t)))
                chosen_strikes.add(float(avail_arr[idx]))

            df_filt = df[df["strike"].isin(chosen_strikes)]

            for _, row in df_filt.iterrows():
                contract_symbol = row.get("contractSymbol") if hasattr(row, "get") else getattr(row, "contractSymbol", None)
                strike = float(row.get("strike") if hasattr(row, "get") else getattr(row, "strike", 0))
                if not contract_symbol or strike <= 0:
                    continue

                bid = _safe_float(row, "bid")
                ask = _safe_float(row, "ask")
                last = _safe_float(row, "lastPrice")
                mid = (bid + ask) / 2.0 if bid is not None and ask is not None else last
                iv = _safe_float(row, "impliedVolatility")
                volume = _safe_float(row, "volume")
                oi = _safe_float(row, "openInterest")

                delta = gamma = theta = vega = None
                if iv is not None and iv > 0:
                    t_years = _years_to_expiry(snapshot_date, exp)
                    computed = compute_greeks(flag=flag, S=spot, K=strike, t=t_years, sigma=iv, r=DEFAULT_R)
                    if computed:
                        delta = computed.get("delta")
                        gamma = computed.get("gamma")
                        theta = computed.get("theta")
                        vega = computed.get("vega")

                all_rows.append({
                    "underlying": symbol,
                    "snapshot_date": snapshot_date.isoformat(),
                    "contract_id": str(contract_symbol),
                    "expiry": exp.isoformat(),
                    "strike": strike,
                    "option_type": opt_type,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "close": last,
                    "iv": iv,
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                    "volume": volume,
                    "open_interest": oi,
                    "source": "yfinance",
                })

    print(f"yfinance subprocess: fetched {len(all_rows)} rows for {symbol}", file=sys.stderr)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Fetch yfinance options for one symbol")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--expiry-targets", default="7,14,30", help="Comma-separated target DTEs")
    parser.add_argument("--sigma-levels", default="-2,-1,0,1,2", help="Comma-separated sigma levels")
    args = parser.parse_args()

    snapshot_date = date.fromisoformat(args.snapshot_date)
    expiry_targets = [int(x) for x in args.expiry_targets.split(",")]
    sigma_levels = [float(x) for x in args.sigma_levels.split(",")]

    rows = fetch_options(args.symbol, snapshot_date, expiry_targets, sigma_levels)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(rows, default=str))
    print(f"Wrote {len(rows)} rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
