import os
import io
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Save output beside this script in a local data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

INDEX_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
]

def fetch_universe_tickers():
    raw_symbols = set()
    sector_mapping = {}

    for url in INDEX_URLS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            df = pd.read_csv(io.StringIO(resp.text))
            
            symbol_col = [c for c in df.columns if "Symbol" in c][0]
            industry_col = [c for c in df.columns if "Industry" in c or "Sector" in c]
            
            for _, row in df.iterrows():
                sym = str(row[symbol_col]).strip()
                if sym and sym != "nan":
                    raw_symbols.add(sym)
                    if industry_col:
                        sector_mapping[sym + ".NS"] = str(row[industry_col[0]]).strip()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    yf_tickers = sorted([f"{s}.NS" for s in raw_symbols])
    print(f"Total unique tickers extracted: {len(yf_tickers)}")
    return yf_tickers, sector_mapping

def download_historical_data(tickers, start_date="2020-01-01", end_date="2026-06-30"):
    print("Downloading historical data...")
    
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        threads=True
    )
    
    adj_close = data["Adj Close"].copy()
    volume = data["Volume"].copy()
    close_raw = data["Close"].copy()

    # Forward fill missing values then drop columns with zero data
    adj_close = adj_close.ffill().dropna(how="all", axis=1)
    volume = volume.fillna(0).dropna(how="all", axis=1)
    close_raw = close_raw.ffill().dropna(how="all", axis=1)

    return adj_close, volume, close_raw

if __name__ == "__main__":
    # Create the directory safely
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tickers, sector_map = fetch_universe_tickers()
    adj_close, volume, close_raw = download_historical_data(tickers)

    # Save to your specified folder
    adj_close.to_csv(os.path.join(OUTPUT_DIR, "adjusted_close.csv"))
    volume.to_csv(os.path.join(OUTPUT_DIR, "volume.csv"))
    close_raw.to_csv(os.path.join(OUTPUT_DIR, "close_unadjusted.csv"))
    pd.Series(sector_map).to_csv(os.path.join(OUTPUT_DIR, "sector_map.csv"), header=["Sector"])
    
    print(f"Files saved successfully to: {OUTPUT_DIR}")