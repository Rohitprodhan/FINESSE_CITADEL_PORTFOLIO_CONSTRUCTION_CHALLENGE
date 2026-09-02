import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIG -- unchanged from backtest_v2.py
# -------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INITIAL_CAPITAL = 1_00_00_000.0
TRANSACTION_FEE_RATE = 0.001
MAX_STOCKS = 10
MAX_SECTOR = 2
CORR_LIMIT = 0.65
RANK_BUFFER = 5
MOMENTUM_WEIGHT = 1.0
SELECTION_VOL_WEIGHT = 0.0

MOMENTUM_LOOKBACK = 250
MOMENTUM_SKIP = 21
SMA_WINDOW = 250
CORR_WINDOW = 60

MAIN_BACKTEST_START = "2021-01-01"
MAIN_BACKTEST_END = "2025-12-31"
OOS_START = "2026-01-01"
OOS_END = "2026-06-30"


# -------------------------------------------------------------------------
# DATA LOADING -- unchanged from backtest_v2.py
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
            "adjusted_close.csv was eligible on every date."
        )
    return adj_close, sector_map, eligibility


# -------------------------------------------------------------------------
# PER-REBALANCE TARGET WEIGHT COMPUTATION -- UNCHANGED strategy logic
# (identical to backtest_v2.py; do not modify)
# -------------------------------------------------------------------------

def compute_target_weights(t_idx, adj_close, sector_map, eligibility, held_stocks):
    lookback_slice = adj_close.iloc[max(0, t_idx - MOMENTUM_LOOKBACK): t_idx]
    if len(lookback_slice) < SMA_WINDOW:
        return {}, 0

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

    ranked = composite_score.index.tolist()
    eligible_for_hold = set(ranked[: MAX_STOCKS + RANK_BUFFER])
    eligible_for_new = set(ranked[:MAX_STOCKS])

    recent_returns = daily_returns.tail(CORR_WINDOW)
    selected_stocks = []
    sector_counts = {}

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
# ACCOUNTING HELPERS -- this is the fix. Pure bookkeeping, no strategy logic.
# -------------------------------------------------------------------------

def _record_buy(positions_tracker, position_episodes, sym, shares_bought, purchase_value,
                 purchase_fee, date):
    """Average-cost accounting on a BUY. Starts a new position episode if this
    is a fresh entry (no existing open position for this ticker)."""
    is_fresh_entry = sym not in positions_tracker or positions_tracker[sym]["shares"] == 0
    if is_fresh_entry:
        positions_tracker[sym] = {"shares": 0, "cost_basis": 0.0}
        position_episodes[sym] = {"realized_pnl": 0.0, "start_date": date}

    positions_tracker[sym]["shares"] += shares_bought
    positions_tracker[sym]["cost_basis"] += purchase_value + purchase_fee


def _record_sale(positions_tracker, position_episodes, closed_positions,
                  realized_transactions, sym, shares_sold, sale_price, fee, date, action):
    """Average-cost accounting on a SELL/TRIM. Realizes P&L on the shares
    sold, reduces cost basis proportionally on the remainder, logs the
    realized transaction, and -- if this brings shares to zero -- closes
    out the position episode with its total realized P&L."""
    pos = positions_tracker[sym]
    old_shares, old_cost_basis = pos["shares"], pos["cost_basis"]

    avg_cost_per_share = old_cost_basis / old_shares if old_shares > 0 else 0.0
    realized_cost_basis = avg_cost_per_share * shares_sold
    sale_value = shares_sold * sale_price
    net_sale_proceeds = sale_value - fee
    realized_pnl = net_sale_proceeds - realized_cost_basis

    pos["cost_basis"] = old_cost_basis - realized_cost_basis
    pos["shares"] = old_shares - shares_sold

    realized_transactions.append({
        "Ticker": sym, "Date": date.strftime("%Y-%m-%d"), "Action": action,
        "Shares_Sold": shares_sold, "Sale_Price": sale_price, "Sale_Value": sale_value,
        "Fee": fee, "Allocated_Cost_Basis": realized_cost_basis, "Realized_PnL": realized_pnl,
    })

    if sym in position_episodes:
        position_episodes[sym]["realized_pnl"] += realized_pnl

    if pos["shares"] <= 0:
        if sym in position_episodes:
            closed_positions.append({
                "Ticker": sym,
                "Start_Date": position_episodes[sym]["start_date"].strftime("%Y-%m-%d"),
                "End_Date": date.strftime("%Y-%m-%d"),
                "Total_PnL": position_episodes[sym]["realized_pnl"],
            })
            del position_episodes[sym]
        del positions_tracker[sym]


# -------------------------------------------------------------------------
# BACKTEST LOOP -- strategy logic (buy/sell decisions, prices, quantities,
# fees, cash) is UNCHANGED from backtest_v2.py. Only the bookkeeping calls
# (_record_buy / _record_sale + the assertions) are new.
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
    positions_tracker = {}       # sym -> {"shares": int, "cost_basis": float}
    position_episodes = {}       # sym -> {"realized_pnl": float, "start_date": Timestamp}
    closed_positions = []        # one row per fully-closed investment episode
    realized_transactions = []   # one row per TRIM or SELL
    trades_log = []              # every order (BUY/TRIM/SELL), unchanged from v2
    portfolio_history = []
    monthly_holdings_log = []
    candidate_pool_sizes = []

    def _check_consistency(sym):
        held = current_holdings.get(sym, 0)
        tracked = positions_tracker.get(sym, {}).get("shares", 0)
        assert held == tracked, (
            f"Share count mismatch for {sym}: current_holdings={held}, "
            f"positions_tracker={tracked}"
        )

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

            # Liquidate anything no longer in the target -- decision logic unchanged
            for sym in list(current_holdings.keys()):
                if sym not in target_weights:
                    sell_price = prices_today[sym]
                    shares_held = current_holdings[sym]
                    sell_proceeds = shares_held * sell_price
                    fee = sell_proceeds * TRANSACTION_FEE_RATE
                    cash += sell_proceeds - fee
                    trades_log.append(
                        {"Date": t.strftime("%Y-%m-%d"), "Ticker": sym, "Action": "SELL",
                         "Shares": shares_held, "Price": sell_price,
                         "Trade_Value": sell_proceeds, "Fee": fee}
                    )
                    if sym in positions_tracker:
                        _record_sale(positions_tracker, position_episodes, closed_positions,
                                      realized_transactions, sym, shares_held, sell_price,
                                      fee, t, "SELL")
                    del current_holdings[sym]
                    _check_consistency(sym)

            # Buy/trim to target -- decision logic (quantities, prices, fees) unchanged
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
                            _record_buy(positions_tracker, position_episodes, sym,
                                        shares_to_buy, cost, fee, t)
                            _check_consistency(sym)
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
                        _record_sale(positions_tracker, position_episodes, closed_positions,
                                      realized_transactions, sym, shares_to_sell,
                                      prices_today[sym], fee, t, "TRIM")
                        if current_holdings[sym] == 0:
                            del current_holdings[sym]
                        _check_consistency(sym)

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

    avg_pool = np.mean(candidate_pool_sizes) if candidate_pool_sizes else float("nan")
    print(f"  Avg eligible candidate pool per rebalance: {avg_pool:.0f} stocks")

    # Unrealized P&L on anything still open at the end of the window
    final_date = backtest_dates[-1] if len(backtest_dates) else None
    final_prices = adj_close.loc[final_date] if final_date is not None else pd.Series(dtype=float)
    unrealized_rows = []
    for sym, pos in positions_tracker.items():
        price = final_prices.get(sym, np.nan)
        if pd.isna(price):
            continue
        market_value = pos["shares"] * price
        unrealized_rows.append({
            "Ticker": sym, "Shares": pos["shares"], "Cost_Basis": pos["cost_basis"],
            "Market_Value": market_value, "Unrealized_PnL": market_value - pos["cost_basis"],
        })

    return {
        "label": label,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "df_results": df_results,
        "df_trades": pd.DataFrame(trades_log),
        "df_realized": pd.DataFrame(realized_transactions),
        "df_closed_positions": pd.DataFrame(closed_positions),
        "df_open_positions": pd.DataFrame(unrealized_rows),
        "monthly_holdings": pd.DataFrame(monthly_holdings_log),
        "cash_end": cash,
    }


# -------------------------------------------------------------------------
# METRICS -- portfolio-level metrics (CAGR, MDD, Sharpe, etc.) unchanged
# -------------------------------------------------------------------------

def compute_portfolio_metrics(run):
    df_results = run["df_results"]
    final_val = df_results["Portfolio_Value"].iloc[-1]
    net_pnl = final_val - INITIAL_CAPITAL
    total_return = (final_val / INITIAL_CAPITAL) - 1.0

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

    df_trades = run["df_trades"]
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
        "Total Rebalance Trades": f"{len(df_trades)}",
        "Total Friction Paid": f"Rs {total_fees:,.2f}",
    }, final_val


def _win_loss_stats(pnl_series):
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    n = len(pnl_series)
    win_rate = (len(wins) / n * 100) if n else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0  # <= 0 or NaN if none
    gain_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss not in (0, None) and not pd.isna(avg_loss) and avg_loss != 0 else np.nan
    gross_profit = wins.sum()
    gross_loss = pnl_series[pnl_series < 0].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.nan
    return {
        "count": n, "wins": len(wins), "losses": len(losses), "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss, "gain_loss_ratio": gain_loss_ratio,
        "profit_factor": profit_factor, "total_pnl": pnl_series.sum(),
    }


def print_diagnostics(run):
    df_realized = run["df_realized"]
    df_closed = run["df_closed_positions"]
    df_open = run["df_open_positions"]

    print("\nREALIZED TRADE STATISTICS")
    print("-" * 40)
    if df_realized.empty:
        print("No realized transactions (no TRIMs or SELLs occurred).")
        tx_stats = None
    else:
        tx_stats = _win_loss_stats(df_realized["Realized_PnL"])
        print(f"Realized transactions       {tx_stats['count']}")
        print(f"Winning transactions        {tx_stats['wins']}")
        print(f"Losing transactions         {tx_stats['losses']}")
        print(f"Win rate                    {tx_stats['win_rate']:.2f}%")
        print(f"Average winning trade       Rs {tx_stats['avg_win']:,.2f}")
        print(f"Average losing trade        Rs {tx_stats['avg_loss']:,.2f}")
        print(f"Gain/Loss ratio             {tx_stats['gain_loss_ratio']:.2f}")
        print(f"Profit factor               {tx_stats['profit_factor']:.2f}")
        print(f"Total realized P&L          Rs {tx_stats['total_pnl']:,.2f}")

    print("\nPOSITION-LEVEL STATISTICS")
    print("-" * 40)
    if df_closed.empty:
        print("No fully-closed positions in this window.")
        pos_stats = None
    else:
        pos_stats = _win_loss_stats(df_closed["Total_PnL"])
        print(f"Completed positions          {pos_stats['count']}")
        print(f"Winning positions             {pos_stats['wins']}")
        print(f"Losing positions              {pos_stats['losses']}")
        print(f"Position win rate            {pos_stats['win_rate']:.2f}%")
        print(f"Average winning position     Rs {pos_stats['avg_win']:,.2f}")
        print(f"Average losing position      Rs {pos_stats['avg_loss']:,.2f}")
        print(f"Position gain/loss ratio     {pos_stats['gain_loss_ratio']:.2f}")
        print(f"Position profit factor       {pos_stats['profit_factor']:.2f}")

    print("\nACCOUNTING RECONCILIATION")
    print("-" * 40)
    total_realized_pnl = df_realized["Realized_PnL"].sum() if not df_realized.empty else 0.0
    unrealized_pnl = df_open["Unrealized_PnL"].sum() if not df_open.empty else 0.0
    final_val = run["df_results"]["Portfolio_Value"].iloc[-1]
    expected_final_val = INITIAL_CAPITAL + total_realized_pnl + unrealized_pnl
    recon_error = final_val - expected_final_val

    print(f"Initial capital              Rs {INITIAL_CAPITAL:,.2f}")
    print(f"Final portfolio value        Rs {final_val:,.2f}")
    print(f"Total realized P&L           Rs {total_realized_pnl:,.2f}")
    print(f"Unrealized P&L (open pos.)   Rs {unrealized_pnl:,.2f}")
    total_fees = run["df_trades"]["Fee"].sum() if not run["df_trades"].empty else 0.0
    print(f"Total fees paid (memo)       Rs {total_fees:,.2f}")
    print(f"Reconciliation error         Rs {recon_error:,.2f}"
          + ("  [OK]" if abs(recon_error) < 1.0 else "  [CHECK THIS]"))

    return tx_stats, pos_stats, recon_error


def print_portfolio_summary(run, metrics):
    print("\n" + "=" * 60)
    print(f"  PORTFOLIO SUMMARY -- {run['label']} "
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
    main_metrics, _ = compute_portfolio_metrics(main_run)
    print_portfolio_summary(main_run, main_metrics)
    print_diagnostics(main_run)

    try:
        oos_run = run_backtest(
            adj_close, sector_map, eligibility, OOS_START, OOS_END, "OUT-OF-SAMPLE 2026 H1"
        )
        if oos_run["df_results"].empty:
            print("\nNo data available for the OOS window yet.")
        else:
            oos_metrics, _ = compute_portfolio_metrics(oos_run)
            print_portfolio_summary(oos_run, oos_metrics)
            print_diagnostics(oos_run)
    except Exception as e:
        print(f"\nOOS run skipped ({e}) -- likely missing 2026 price data.")

    # Save outputs
    pd.DataFrame(list(main_metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(DATA_DIR, "performance_summary.csv"), index=False
    )
    main_run["df_results"].to_csv(os.path.join(DATA_DIR, "backtest_curve.csv"))
    main_run["df_trades"].to_csv(os.path.join(DATA_DIR, "trades_log.csv"), index=False)
    main_run["df_realized"].to_csv(os.path.join(DATA_DIR, "realized_transactions.csv"), index=False)
    main_run["df_closed_positions"].to_csv(os.path.join(DATA_DIR, "closed_positions.csv"), index=False)
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

    #plotting the equity curve for the OOS run if it exists
    if 'oos_run' in locals() and not oos_run["df_results"].empty:
        plt.figure(figsize=(12, 6))
        plt.plot(
            pd.to_datetime(oos_run["df_results"].index),
            oos_run["df_results"]["Portfolio_Value"] / 1_00_00_000,
            label="OOS Portfolio (Rs Cr)", color="#aa5500", lw=2,
        )
        plt.title("Out-of-Sample Portfolio Growth 2026 H1", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (Rs Crores)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "oos_equity_curve.png"), dpi=300)

    print(f"\nFiles written to {DATA_DIR}/: performance_summary.csv, backtest_curve.csv, "
          "trades_log.csv, realized_transactions.csv, closed_positions.csv, "
          "monthly_holdings.csv, equity_curve.png")


if __name__ == "__main__":
    main()
