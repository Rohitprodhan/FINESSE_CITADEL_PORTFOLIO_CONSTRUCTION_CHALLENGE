# Finesse x Citadel Portfolio Challenge — Round 2 Submission

Systematic, rule-based equity portfolio strategy for the Nifty 100 / Nifty Midcap 100 /
Nifty Smallcap 100 universe. Selects up to 10 stocks each month using a momentum
ranking with a trend filter, sector cap, and correlation-based diversification screen,
sized by inverse-volatility (risk-parity) weighting, backtested 2021–2025 with an
out-of-sample stress test on Jan–Jun 2026.

## Repository structure

```
.
├── data_loader.py      # Pulls price/volume history and index constituents
├── backtest.py         # Runs the strategy, backtest, and out-of-sample test
├── data/                # Input data + generated outputs (see "Data" below)
├── requirements.txt
├── report.pdf
└── README.md
```

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
matplotlib
yfinance
requests
```

## How to run

**Data is already included in `data/`** (see below), so you can run the backtest
directly without re-fetching anything:

```bash
python backtest.py
```

This prints the 2021–2025 backtest summary, the out-of-sample (Jan–Jun 2026) summary,
realized trade / position-level diagnostics, and an accounting reconciliation check to
the console, and writes result files to `data/` (see "Outputs" below).

**To re-fetch the source data from scratch** (optional — only needed if you want to
refresh prices or verify the data pipeline independently):

```bash
python data_loader.py
```

This downloads current index constituents and 2020–2026 daily price/volume history and
overwrites the input files in `data/`. Note: this can take several minutes and depends
on niftyindices.com and Yahoo Finance being reachable and not rate-limiting the run.

## Data

### Inputs (produced by `data_loader.py`, included in `data/`)

| File | Description | Source |
|---|---|---|
| `adjusted_close.csv` | Daily split/dividend-adjusted close price, wide format (dates × tickers) | Yahoo Finance (`yfinance`), `Adj Close` field |
| `close_unadjusted.csv` | Daily raw close price, same shape | Yahoo Finance, `Close` field |
| `volume.csv` | Daily traded volume, same shape | Yahoo Finance, `Volume` field |
| `sector_map.csv` | Ticker → industry/sector classification | niftyindices.com index constituent lists |

- **Universe:** current constituents of the Nifty 100, Nifty Midcap 100, and Nifty
  Smallcap 100 as published on niftyindices.com at the time of the data pull, mapped to
  NSE tickers (`.NS` suffix) for Yahoo Finance.
- **Frequency / period:** daily, 2020-01-01 to 2026-06-30 (padded before the 2021-01-01
  backtest start so momentum/trend factors have a full lookback window from day one).
- **Cleaning:** forward-filled for short data gaps; tickers with no data at all are
  dropped. `adjusted_close.csv` is the field the strategy trades on throughout.

### Outputs (produced by `backtest.py`, included in `data/`)

| File | Description |
|---|---|
| `performance_summary.csv` | Headline portfolio metrics for the 2021–2025 backtest (return, CAGR, MDD, Sharpe, volatility, trade count, fees) |
| `backtest_curve.csv` | Daily portfolio value, cash, holdings value, and daily return |
| `trades_log.csv` | Every BUY/TRIM/SELL order: date, ticker, action, shares, price, value, fee |
| `realized_transactions.csv` | Every TRIM/SELL as a realized transaction with allocated cost basis and realized P&L (average-cost accounting) |
| `closed_positions.csv` | One row per fully-closed investment episode (first BUY through final SELL), with total realized P&L |
| `monthly_holdings.csv` | Target portfolio weights at every monthly rebalance date |
| `equity_curve.png` | Portfolio value chart, 2021–2025 |
| `oos_equity_curve.png` | Portfolio value chart, out-of-sample Jan–Jun 2026 window |

The console output additionally reports realized-transaction statistics, position-level
statistics (win rate, gain/loss ratio, profit factor for each), and an accounting
reconciliation check confirming `initial capital + realized P&L + unrealized P&L =
final portfolio value` for both the main backtest and the out-of-sample run.

## Methodology summary

- **Selection:** stocks are ranked by 12-minus-1-month price momentum (skipping the
  most recent ~21 trading days), after passing a 200/250-day SMA trend filter and basic
  liquidity/data-completeness checks.
- **Diversification:** a sector cap (max 2 stocks per industry) and a rolling 60-day
  pairwise correlation screen (max 0.65) are applied greedily down the ranked list.
- **Weighting:** inverse-volatility (risk-parity) sizing across the selected names.
- **Rebalancing:** monthly, on the first trading day of the month. A held stock is only
  dropped once it falls outside the top 15 ranked names (a small buffer above the
  top-10 cutoff used for new entrants), which reduces unnecessary turnover from names
  oscillating right at the selection boundary.
- **Costs:** 0.1% transaction fee applied on every buy, sell, and trim, deducted from
  cash at execution.
- **Starting capital:** ₹1,00,00,000.
- **Backtest window:** 2021-01-01 to 2025-12-31. **Out-of-sample stress test:**
  2026-01-01 to 2026-06-30, run with the identical, unmodified strategy code.

Full factor definitions, formulas, and rationale are in the accompanying report.

## Known limitations / assumptions (disclosed)

- **Universe membership is not point-in-time.** `data_loader.py` fetches the *current*
  Nifty 100/Midcap100/Smallcap100 constituent lists and applies that fixed set across
  the entire 2021–2026 backtest. Stocks that were added to or dropped from these
  indices during the period are not reflected at the date they actually joined or left.
  This is a survivorship-bias risk: the universe is implicitly biased toward stocks
  that are still index members today. We did not have a historical constituent-by-date
  source available in the time available; this is disclosed here and in the report
  rather than left implicit.
- **Fractional shares are not modeled.** Share quantities are floored to whole shares
  at each trade, which leaves small amounts of cash undeployed between rebalances.
- **No slippage or market-impact model beyond the flat 0.1% fee.** Execution is assumed
  at the same day's adjusted close price for the full order size.
- **Sector classification** is taken as of the current constituent-list pull, not
  point-in-time, for the same reason as universe membership above.
- **Corporate actions:** we rely on Yahoo Finance's `Adj Close` field for
  split/dividend adjustment rather than adjusting manually; we spot-checked but did not
  exhaustively audit every ticker for adjustment errors.
