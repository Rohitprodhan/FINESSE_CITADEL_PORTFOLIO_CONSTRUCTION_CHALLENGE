"""
backtest_v2.py
--------------
Revised systematic equity backtest -- builds on stock_selection.py, addressing:

  1. POINT-IN-TIME UNIVERSE: optional data/universe_eligibility.csv (dates x
     tickers boolean panel) so index membership can reflect what was actually
     eligible at each rebalance date, not today's constituent list applied
     retroactively. Falls back to "always eligible" with a loud warning if
     the file isn't present -- so the survivorship-bias risk stays visible
     instead of silently defaulting.

  2. DECOUPLED VOLATILITY SIGNAL: inverse-volatility was driving both the
     SELECTION score (40% weight) and the WEIGHTING rule (100%) in the
     original script -- the same signal doing double duty, likely biasing
     the portfolio toward calmer large/mid-caps. SELECTION_VOL_WEIGHT below
     defaults to 0.0 (selection = momentum only; weighting still uses
     inverse-vol). Set it back to 0.40 to reproduce the original design and
     compare.

  3. CORR_LIMIT FIXED to 0.65 to match what the report documents (the
     uploaded script had 0.9, which barely filters anything -- see chat).

  4. TURNOVER HYSTERESIS: a stock already held is only sold if it drops
     outside the top (MAX_STOCKS + RANK_BUFFER) rather than outside the top
     MAX_STOCKS -- cuts churn from names oscillating right at the cutoff.
     New entrants still need to be in the top MAX_STOCKS to get bought.

  5. SHARED, REUSABLE run_backtest(): the exact same function drives both
     the 2021-2025 backtest and the Jan-Jun 2026 out-of-sample stress test,
     so the OOS number is a real measurement, not a description.

  6. CAGR bug fix: original script hardcoded `** (1/5)`, which silently
     breaks for any window that isn't exactly 5 years (e.g. the 6-month OOS
     run). Now computed from actual elapsed calendar days.

Everything else (sector cap, SMA200 trend filter, monthly rebalance, greedy
correlation screen, inverse-vol weighting) is unchanged from your working
version.

Run:
    python backtest_v2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------

DATA_DIR = "data"
INITIAL_CAPITAL = 1_00_00_000.0
TRANSACTION_FEE_RATE = 0.001
MAX_STOCKS = 10
MAX_SECTOR = 2
CORR_LIMIT = 0.65          # FIXED -- was 0.9 in the uploaded script (see docstring)
RANK_BUFFER = 5            # hold until a name falls outside top (MAX_STOCKS + this)
MOMENTUM_WEIGHT = 1.0
SELECTION_VOL_WEIGHT = 0.0  # 0.0 = decoupled (recommended); 0.40 = original double-counted design

MOMENTUM_LOOKBACK = 378
MOMENTUM_SKIP = 21
SMA_WINDOW = 200
CORR_WINDOW = 60

MAIN_BACKTEST_START = "2021-01-01"
MAIN_BACKTEST_END = "2025-12-31"
OOS_START = "2026-01-01"
OOS_END = "2026-06-30"


# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------

def load_data():
    adj_close = pd.read_csv(
        os.path.join(DATA_DIR, "adjusted_close.csv"), index_col=0, parse_dates=True
    )
    sector_map = pd.read_csv(
        os.path.join(DATA_DIR, "sector_map.csv"), index_col=0
    )["Sector"].to_dict()

    elig_path = os.path.join(DATA_DIR, "universe_eligibility.csv")
    if os.path.exists(elig_path):
        eligibility = pd.read_csv(elig_path, index_col=0, parse_dates=True)
        eligibility = eligibility.reindex(index=adj_close.index, columns=adj_close.columns)
        eligibility = eligibility.fillna(False).astype(bool)
        print("Loaded point-in-time universe eligibility from universe_eligibility.csv")
    else:
        eligibility = pd.DataFrame(True, index=adj_close.index, columns=adj_close.columns)
        print(
            "WARNING: no data/universe_eligibility.csv found -- assuming every ticker in "
            "adjusted_close.csv was eligible on every date. This is a survivorship-bias "
            "risk if that column set reflects TODAY's index membership. Disclose this "
            "assumption explicitly in the report if you don't fix it before submission."
        )
    return adj_close, sector_map, eligibility


# -------------------------------------------------------------------------
# PER-REBALANCE TARGET WEIGHT COMPUTATION (shared by both backtest windows)
# -------------------------------------------------------------------------

def compute_target_weights(t_idx, adj_close, sector_map, eligibility, held_stocks):
    """Compute target portfolio weights as of trading-day index t_idx.

    held_stocks: set of tickers currently held, used for the rank-hysteresis
    buffer (point 4) -- held names get more room before being dropped.
    """
    lookback_slice = adj_close.iloc[max(0, t_idx - MOMENTUM_LOOKBACK): t_idx]
    if len(lookback_slice) < SMA_WINDOW:
        return {}, 0  # not enough history yet

    p_t = lookback_slice.iloc[-1]
    p_t21 = lookback_slice.iloc[-MOMENTUM_SKIP]
    p_t_start = lookback_slice.iloc[0]

    momentum = (p_t21 / p_t_start) - 1.0
    daily_returns = lookback_slice.pct_change()
    valid_counts = daily_returns.count()
    volatility = daily_returns.std() * np.sqrt(252)
    stability = 1.0 / (volatility + 1e-6)

    sma = lookback_slice.tail(SMA_WINDOW).mean()
    trend_passed = p_t > sma

    today_eligible = eligibility.iloc[t_idx] if t_idx < len(eligibility) else pd.Series(
        True, index=adj_close.columns
    )

    valid = (
        (valid_counts >= 180)
        & trend_passed
        & momentum.notna()
        & volatility.notna()
        & (volatility > 0.05)
        & (volatility < 1.0)
        & p_t.notna()
        & today_eligible
    )
    candidate_pool = valid[valid].index.tolist()
    if len(candidate_pool) < 5:
        return {}, len(candidate_pool)

    z_mom = (momentum[candidate_pool] - momentum[candidate_pool].mean()) / (
        momentum[candidate_pool].std() + 1e-6
    )
    z_stab = (stability[candidate_pool] - stability[candidate_pool].mean()) / (
        stability[candidate_pool].std() + 1e-6
    )
    composite_score = (
        MOMENTUM_WEIGHT * z_mom + SELECTION_VOL_WEIGHT * z_stab
    ).sort_values(ascending=False)

    # Rank-hysteresis buffer: a currently-held stock is allowed to rank up to
    # (MAX_STOCKS + RANK_BUFFER) and still be kept; a NEW entrant still needs
    # to be in the top MAX_STOCKS. This is what cuts unnecessary churn.
    ranked = composite_score.index.tolist()
    eligible_for_hold = set(ranked[: MAX_STOCKS + RANK_BUFFER])
    eligible_for_new = set(ranked[:MAX_STOCKS])

    recent_returns = daily_returns.tail(CORR_WINDOW)
    selected_stocks = []
    sector_counts = {}

    # Pass 1: keep already-held names that are still within the hold buffer,
    # in their score order, subject to the same sector/correlation rules.
    for sym in ranked:
        if len(selected_stocks) >= MAX_STOCKS:
            break
        if sym not in held_stocks or sym not in eligible_for_hold:
            continue
        sec = sector_map.get(sym, "General")
        if sector_counts.get(sec, 0) >= MAX_SECTOR:
            continue
        if selected_stocks:
            corrs = [
                recent_returns[sym].corr(recent_returns[e])
                for e in selected_stocks
                if e in recent_returns
            ]
            corrs = [c for c in corrs if not np.isnan(c)]
            if corrs and max(corrs) > CORR_LIMIT:
                continue
        selected_stocks.append(sym)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # Pass 2: fill remaining slots from the top-MAX_STOCKS ranked list
    # (covers both fresh entrants and any held names that didn't make pass 1).
    for sym in ranked:
        if len(selected_stocks) >= MAX_STOCKS:
            break
        if sym in selected_stocks or sym not in eligible_for_new:
            continue
        sec = sector_map.get(sym, "General")
        if sector_counts.get(sec, 0) >= MAX_SECTOR:
            continue
        if selected_stocks:
            corrs = [
                recent_returns[sym].corr(recent_returns[e])
                for e in selected_stocks
                if e in recent_returns
            ]
            corrs = [c for c in corrs if not np.isnan(c)]
            if corrs and max(corrs) > CORR_LIMIT:
                continue
        selected_stocks.append(sym)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    if not selected_stocks:
        return {}, len(candidate_pool)

    inv_vol = 1.0 / volatility[selected_stocks]
    target_weights = (inv_vol / inv_vol.sum()).to_dict()
    return target_weights, len(candidate_pool)


# -------------------------------------------------------------------------
# BACKTEST LOOP (shared by main backtest and OOS stress test)
# -------------------------------------------------------------------------

def run_backtest(adj_close, sector_map, eligibility, start, end, label):
    print(f"\nRunning backtest [{label}]: {start} -> {end}")

    all_dates = adj_close.index
    backtest_dates = adj_close.loc[start:end].index
    rebalance_dates_raw = adj_close.loc[start:end].resample("MS").first().index
    rebalance_dates = [
        adj_close.loc[d:].index[0] for d in rebalance_dates_raw if len(adj_close.loc[d:]) > 0
    ]

    cash = INITIAL_CAPITAL
    current_holdings = {}
    positions_tracker = {}
    closed_trades = []
    trades_log = []
    portfolio_history = []
    monthly_holdings_log = []
    candidate_pool_sizes = []

    for t in backtest_dates:
        t_idx = all_dates.get_loc(t)
        prices_today = adj_close.iloc[t_idx]

        if t in rebalance_dates:
            target_weights, pool_size = compute_target_weights(
                t_idx, adj_close, sector_map, eligibility, set(current_holdings.keys())
            )
            candidate_pool_sizes.append(pool_size)

            for sym, weight in target_weights.items():
                monthly_holdings_log.append(
                    {"Date": t.strftime("%Y-%m-%d"), "Ticker": sym, "Target_Weight": weight}
                )

            # Liquidate anything no longer in the target
            for sym in list(current_holdings.keys()):
                if sym not in target_weights:
                    sell_price = prices_today[sym]
                    sell_proceeds = current_holdings[sym] * sell_price
                    fee = sell_proceeds * TRANSACTION_FEE_RATE
                    cash += sell_proceeds - fee
                    trades_log.append(
                        {"Date": t.strftime("%Y-%m-%d"), "Ticker": sym, "Action": "SELL",
                         "Shares": current_holdings[sym], "Price": sell_price,
                         "Trade_Value": sell_proceeds, "Fee": fee}
                    )
                    if sym in positions_tracker:
                        pnl = (sell_proceeds - fee) - positions_tracker[sym]["cost"]
                        closed_trades.append({"Ticker": sym, "PnL": pnl})
                        del positions_tracker[sym]
                    del current_holdings[sym]

            # Buy/trim to target
            total_value = cash + sum(
                current_holdings[s] * prices_today[s]
                for s in current_holdings if not np.isnan(prices_today[s])
            )
            for sym, weight in target_weights.items():
                if np.isnan(prices_today[sym]) or prices_today[sym] <= 0:
                    continue
                target_val = total_value * weight
                curr_val = current_holdings.get(sym, 0) * prices_today[sym]
                diff = target_val - curr_val

                if diff > 0:
                    provisional_fee = diff * TRANSACTION_FEE_RATE
                    if cash >= diff + provisional_fee:
                        shares_to_buy = int(diff / prices_today[sym])
                        if shares_to_buy > 0:
                            cost = shares_to_buy * prices_today[sym]
                            fee = cost * TRANSACTION_FEE_RATE
                            cash -= cost + fee
                            current_holdings[sym] = current_holdings.get(sym, 0) + shares_to_buy
                            trades_log.append(
                                {"Date": t.strftime("%Y-%m-%d"), "Ticker": sym, "Action": "BUY",
                                 "Shares": shares_to_buy, "Price": prices_today[sym],
                                 "Trade_Value": cost, "Fee": fee}
                            )
                            if sym not in positions_tracker:
                                positions_tracker[sym] = {"cost": cost + fee}
                            else:
                                positions_tracker[sym]["cost"] += cost + fee
                elif diff < 0:
                    shares_to_sell = int(abs(diff) / prices_today[sym])
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * prices_today[sym]
                        fee = proceeds * TRANSACTION_FEE_RATE
                        cash += proceeds - fee
                        current_holdings[sym] -= shares_to_sell
                        trades_log.append(
                            {"Date": t.strftime("%Y-%m-%d"), "Ticker": sym, "Action": "TRIM",
                             "Shares": shares_to_sell, "Price": prices_today[sym],
                             "Trade_Value": proceeds, "Fee": fee}
                        )

        holdings_value = sum(
            current_holdings[sym] * prices_today[sym]
            for sym in current_holdings if not np.isnan(prices_today[sym])
        )
        portfolio_history.append(
            {"Date": t.strftime("%Y-%m-%d"), "Portfolio_Value": cash + holdings_value,
             "Cash": cash, "Holdings_Value": holdings_value}
        )

    df_results = pd.DataFrame(portfolio_history).set_index("Date")
    df_results["Daily_Return"] = df_results["Portfolio_Value"].pct_change().fillna(0)
    df_trades = pd.DataFrame(trades_log)
    df_closed = pd.DataFrame(closed_trades)

    avg_pool = np.mean(candidate_pool_sizes) if candidate_pool_sizes else float("nan")
    print(f"  Avg eligible candidate pool per rebalance: {avg_pool:.0f} stocks")

    return {
        "label": label,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "df_results": df_results,
        "df_trades": df_trades,
        "df_closed": df_closed,
        "monthly_holdings": pd.DataFrame(monthly_holdings_log),
    }


# -------------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------------

def compute_metrics(run):
    df_results, df_trades, df_closed = run["df_results"], run["df_trades"], run["df_closed"]

    final_val = df_results["Portfolio_Value"].iloc[-1]
    net_pnl = final_val - INITIAL_CAPITAL
    total_return = (final_val / INITIAL_CAPITAL) - 1.0

    # FIXED: CAGR from actual elapsed days, not a hardcoded 5-year exponent
    elapsed_days = (
        pd.to_datetime(df_results.index[-1]) - pd.to_datetime(df_results.index[0])
    ).days
    years = max(elapsed_days / 365.25, 1e-6)
    cagr = (final_val / INITIAL_CAPITAL) ** (1.0 / years) - 1.0

    rolling_max = df_results["Portfolio_Value"].cummax()
    drawdown = (df_results["Portfolio_Value"] - rolling_max) / rolling_max
    mdd = drawdown.min()

    daily_std = df_results["Daily_Return"].std()
    sharpe = (df_results["Daily_Return"].mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0
    annualized_vol = daily_std * np.sqrt(252)

    if not df_closed.empty:
        wins = df_closed[df_closed["PnL"] > 0]
        losses = df_closed[df_closed["PnL"] <= 0]
        accuracy = (len(wins) / len(df_closed)) * 100
        avg_win = wins["PnL"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["PnL"].mean()) if len(losses) > 0 else np.nan
        gain_to_loss = avg_win / avg_loss if avg_loss else np.nan
    else:
        accuracy, gain_to_loss = 0.0, np.nan

    total_fees = df_trades["Fee"].sum() if not df_trades.empty else 0.0

    return {
        "Years covered": round(years, 2),
        "Starting Capital": f"Rs {INITIAL_CAPITAL:,.2f}",
        "Final Portfolio Value": f"Rs {final_val:,.2f}",
        "Total Net PnL": f"Rs {net_pnl:,.2f}",
        "Absolute Return": f"{total_return * 100:.2f}%",
        "Annualized Return (CAGR)": f"{cagr * 100:.2f}%",
        "Maximum Drawdown (MDD)": f"{mdd * 100:.2f}%",
        "Sharpe Ratio (Rf=0%)": f"{sharpe:.2f}",
        "Annualized Volatility": f"{annualized_vol * 100:.2f}%",
        "Trade Accuracy (Win Rate)": f"{accuracy:.2f}%",
        "Gain-to-Loss Ratio": f"{gain_to_loss:.2f}",
        "Total Rebalance Trades": f"{len(df_trades)}",
        "Total Friction Paid": f"Rs {total_fees:,.2f}",
    }


def print_metrics(run, metrics):
    print("\n" + "=" * 60)
    print(f"  BACKTEST SUMMARY -- {run['label']} "
          f"({run['start'].date()} -> {run['end'].date()})")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:<28}{v}")
    print("=" * 60)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    adj_close, sector_map, eligibility = load_data()

    main_run = run_backtest(
        adj_close, sector_map, eligibility, MAIN_BACKTEST_START, MAIN_BACKTEST_END, "MAIN 2021-2025"
    )
    main_metrics = compute_metrics(main_run)
    print_metrics(main_run, main_metrics)

    # Out-of-sample stress test -- identical function, fresh capital, no
    # parameter changes. If your price data doesn't extend to mid-2026 yet,
    # this will just fail cleanly on the date slice -- pull more data first.
    try:
        oos_run = run_backtest(
            adj_close, sector_map, eligibility, OOS_START, OOS_END, "OUT-OF-SAMPLE 2026 H1"
        )
        if oos_run["df_results"].empty:
            print("\nNo data available for the OOS window yet -- extend your price history to include Jan-Jun 2026.")
        else:
            oos_metrics = compute_metrics(oos_run)
            print_metrics(oos_run, oos_metrics)
    except Exception as e:
        print(f"\nOOS run skipped ({e}) -- likely missing 2026 price data.")

    # Save main-backtest outputs
    pd.DataFrame(list(main_metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(DATA_DIR, "performance_summary.csv"), index=False
    )
    main_run["df_results"].to_csv(os.path.join(DATA_DIR, "backtest_curve.csv"))
    main_run["df_trades"].to_csv(os.path.join(DATA_DIR, "trades_log.csv"), index=False)
    main_run["monthly_holdings"].to_csv(os.path.join(DATA_DIR, "monthly_holdings.csv"), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(
        pd.to_datetime(main_run["df_results"].index),
        main_run["df_results"]["Portfolio_Value"] / 1_00_00_000,
        label="Strategy Portfolio (Rs Cr)", color="#0055aa", lw=2,
    )
    plt.title("Portfolio Growth 2021-2025", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Rs Crores)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "equity_curve.png"), dpi=300)

    print(f"\nFiles written to {DATA_DIR}/: performance_summary.csv, backtest_curve.csv, "
          "trades_log.csv, monthly_holdings.csv, equity_curve.png")


if __name__ == "__main__":
    main()
