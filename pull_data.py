"""
pull_data.py
------------
Data pull script for the Finesse x Citadel Round 2 portfolio challenge.

What this does:
  1. Loads your stock universe (a list of NSE tickers).
  2. Downloads daily OHLCV price history for 2020-06-01 -> 2026-06-30
     (padded before 2021 so momentum/rolling factors have lookback data,
     and through 2026-06-30 to cover the out-of-sample stress window).
  3. Downloads a snapshot of basic fundamentals per ticker (current only --
     see the LIMITATIONS note below).
  4. Saves everything to ./data/ as CSV files ready for the backtest engine.

Requirements:
    pip install yfinance pandas --break-system-packages   (or in a venv)

Run:
    python pull_data.py

-------------------------------------------------------------------------
IMPORTANT LIMITATIONS -- read before you rely on this for the backtest
-------------------------------------------------------------------------
1. UNIVERSE / CONSTITUENT LIST:
   This script does NOT know which stocks were in the Nifty 100 / Midcap 100
   / Smallcap 100 at each historical rebalance date -- index membership
   changes over time, and using today's constituent list for the whole
   2021-2025 backtest introduces survivorship bias (you'd only be selecting
   from stocks that are winners *today*).
   -> Get historical constituent lists (ideally as of each quarter-end) from
      niftyindices.com (Historical Data > Index constituents) if you have
      time. If not, at minimum disclose the current-constituent-list
      shortcut explicitly in your report's Limitations section -- evaluators
      specifically ask about this kind of assumption.

2. FUNDAMENTALS ARE A SNAPSHOT, NOT A TIME SERIES:
   yfinance's `.info` only gives you *current* fundamentals (ROE, D/E, PE,
   etc.), not what they were in, say, 2022. If your factor model uses
   fundamentals point-in-time, you need a proper point-in-time source
   (e.g. screener.in export, Refinitiv/Bloomberg if your team has access,
   or a paid fundamentals API). Using today's fundamentals throughout the
   backtest is a lookahead bias -- flag this in your report if it's what
   you end up doing under time pressure.

3. yfinance reliability: Yahoo occasionally rate-limits or returns gaps for
   NSE tickers. The script retries and logs failures to failed_tickers.txt
   so you can re-run just those.
"""

import time
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

# -------------------------------------------------------------------------
# CONFIG -- edit this section
# -------------------------------------------------------------------------

# Fill this in with your actual universe (NSE symbols, WITHOUT the .NS
# suffix -- the script adds it). This is a small starter sample so you can
# test the pipeline; replace with your full Nifty 100/Midcap100/Smallcap100
# list (pull the constituent CSVs from niftyindices.com).
TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "LT", "SBIN", "BAJFINANCE",
]

START_DATE = "2020-06-01"   # padded before 2021 for rolling-window factors
END_DATE = "2026-06-30"     # covers backtest + out-of-sample stress window
BENCHMARK_TICKER = "^CRSLDX"  # Nifty 500 index (use "^CNX100" for Nifty 100)

OUT_DIR = Path("data")
RETRIES = 3
RETRY_DELAY_SEC = 5

# -------------------------------------------------------------------------


def fetch_price_history(ticker: str) -> pd.DataFrame | None:
    """Download daily OHLCV for one ticker with retries."""
    symbol = ticker if ticker.startswith("^") else f"{ticker}.NS"
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(
                symbol,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                raise ValueError("empty dataframe returned")
            df.index.name = "date"
            return df
        except Exception as e:
            print(f"  [{ticker}] attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    return None


def fetch_fundamentals_snapshot(ticker: str) -> dict:
    """Grab a current fundamentals snapshot. See LIMITATIONS in module docstring."""
    symbol = f"{ticker}.NS"
    try:
        info = yf.Ticker(symbol).info
        return {
            "ticker": ticker,
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception as e:
        print(f"  [{ticker}] fundamentals fetch failed: {e}")
        return {"ticker": ticker}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    prices_dir = OUT_DIR / "prices"
    prices_dir.mkdir(exist_ok=True)

    failed = []
    fundamentals_rows = []

    print(f"Pulling price history for {len(TICKERS)} tickers "
          f"({START_DATE} -> {END_DATE})...")
    for i, ticker in enumerate(TICKERS, 1):
        print(f"[{i}/{len(TICKERS)}] {ticker}")
        df = fetch_price_history(ticker)
        if df is None:
            failed.append(ticker)
            continue
        df.to_csv(prices_dir / f"{ticker}.csv")

        fundamentals_rows.append(fetch_fundamentals_snapshot(ticker))
        time.sleep(0.5)  # be polite to the API

    # Benchmark
    print(f"\nPulling benchmark {BENCHMARK_TICKER}...")
    bench_df = fetch_price_history(BENCHMARK_TICKER)
    if bench_df is not None:
        bench_df.to_csv(OUT_DIR / "benchmark.csv")
    else:
        print("  WARNING: benchmark pull failed -- try again or pick a different index ticker.")

    # Fundamentals snapshot table
    if fundamentals_rows:
        pd.DataFrame(fundamentals_rows).to_csv(
            OUT_DIR / "fundamentals_snapshot.csv", index=False
        )

    # Log any failures so you can re-run just those
    if failed:
        (OUT_DIR / "failed_tickers.txt").write_text("\n".join(failed))
        print(f"\n{len(failed)} tickers failed: {failed}")
        print("Logged to data/failed_tickers.txt -- re-run after checking symbols.")
    else:
        print("\nAll tickers pulled successfully.")

    # Small manifest so you (and the README) know what's in data/ and when
    manifest = {
        "tickers": TICKERS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "benchmark": BENCHMARK_TICKER,
        "pulled_at": pd.Timestamp.now().isoformat(),
        "failed_tickers": failed,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Data saved under {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()