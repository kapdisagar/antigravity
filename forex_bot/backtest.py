"""
Backtest Runner — tests the Trident Pattern strategy on historical MT5 data.

Fetches historical 30M candles from MT5, runs the pattern detection logic,
simulates trades, and reports performance metrics.

Usage:
    python backtest.py
    python backtest.py --symbol XAUUSD --days 180
"""

import sys
import io
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import numpy as np

import config
import mt5_connector as mt5c
from indicators import calculate_emas, are_emas_stacked, get_200ema_bias
from fvg_detector import find_fvgs
from trident_pattern import validate_trident_pattern
from time_filter import is_in_kill_zone
from logger import setup_logger

log = setup_logger("Backtest")


# ─── Trade Record ──────────────────────────────────────────────────────────────
@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    stop_loss: float
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl_pips: float = 0.0
    rr_ratio: float = 0.0
    result: str = "OPEN"    # "WIN", "LOSS", "OPEN"
    exit_reason: str = ""


# ─── Backtest Engine ───────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pips: float = 0.0
    avg_rr: float = 0.0
    max_drawdown_pips: float = 0.0
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)


def get_pip_value(symbol: str) -> float:
    """Get pip value for a symbol. Approximation for backtest."""
    symbol_upper = symbol.upper()
    if "JPY" in symbol_upper:
        return 0.01
    elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return 0.1
    else:
        return 0.0001


def simulate_trade_exit(df_30m: pd.DataFrame, df_daily: pd.DataFrame,
                        trade: BacktestTrade, entry_bar_idx: int,
                        pip_value: float) -> BacktestTrade:
    """
    Simulate forward from entry to find the exit point.
    
    Exit conditions:
    1. Price hits stop loss (candle close for Gold, hard SL for others)
    2. Daily EMAs break stacking
    3. Significant opposing candle on daily chart
    4. Max hold: 20 daily candles (safety net)
    """
    is_gold = trade.symbol.upper() in ["XAUUSD", "GOLD"]
    risk_pips = abs(trade.entry_price - trade.stop_loss) / pip_value

    # Scan forward through 30M candles
    max_bars = min(len(df_30m), entry_bar_idx + 960)  # ~20 days of 30M bars

    for i in range(entry_bar_idx + 1, max_bars):
        candle = df_30m.iloc[i]

        if trade.direction == "BUY":
            # Check stop loss hit
            if not is_gold:
                if candle["low"] <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = candle["time"]
                    trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
                    trade.result = "LOSS"
                    trade.exit_reason = "Stop loss hit"
                    return trade
            else:
                # Gold: candle CLOSE filter
                if candle["close"] <= trade.stop_loss:
                    trade.exit_price = candle["close"]
                    trade.exit_time = candle["time"]
                    trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
                    trade.result = "LOSS"
                    trade.exit_reason = "Gold candle close filter"
                    return trade

        elif trade.direction == "SELL":
            if not is_gold:
                if candle["high"] >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = candle["time"]
                    trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
                    trade.result = "LOSS"
                    trade.exit_reason = "Stop loss hit"
                    return trade
            else:
                if candle["close"] >= trade.stop_loss:
                    trade.exit_price = candle["close"]
                    trade.exit_time = candle["time"]
                    trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
                    trade.result = "LOSS"
                    trade.exit_reason = "Gold candle close filter"
                    return trade

        # Check daily exit conditions every ~48 bars (1 day of 30M)
        if (i - entry_bar_idx) % 48 == 0 and not df_daily.empty:
            # Find the corresponding daily candle
            candle_date = candle["time"].date() if hasattr(candle["time"], "date") else candle["time"]
            daily_up_to = df_daily[df_daily["time"].dt.date <= candle_date] if hasattr(df_daily["time"].dt, "date") else df_daily

            if len(daily_up_to) >= 5:
                daily_subset = calculate_emas(daily_up_to.copy(), config.EMA_FAST_PERIODS)

                if trade.direction == "BUY":
                    if not are_emas_stacked(daily_subset, "long", -1):
                        was_stacked = are_emas_stacked(daily_subset, "long", -2)
                        if was_stacked:
                            trade.exit_price = candle["close"]
                            trade.exit_time = candle["time"]
                            trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
                            trade.result = "WIN" if trade.pnl_pips > 0 else "LOSS"
                            trade.exit_reason = "Daily EMA unstack"
                            return trade

                elif trade.direction == "SELL":
                    if not are_emas_stacked(daily_subset, "short", -1):
                        was_stacked = are_emas_stacked(daily_subset, "short", -2)
                        if was_stacked:
                            trade.exit_price = candle["close"]
                            trade.exit_time = candle["time"]
                            trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
                            trade.result = "WIN" if trade.pnl_pips > 0 else "LOSS"
                            trade.exit_reason = "Daily EMA unstack"
                            return trade

    # Max hold reached — close at last candle
    if max_bars < len(df_30m):
        last_candle = df_30m.iloc[max_bars - 1]
    else:
        last_candle = df_30m.iloc[-1]

    trade.exit_price = last_candle["close"]
    trade.exit_time = last_candle["time"]
    if trade.direction == "BUY":
        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_value
    else:
        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_value
    trade.result = "WIN" if trade.pnl_pips > 0 else "LOSS"
    trade.exit_reason = "Max hold period"
    return trade


def backtest_symbol(symbol: str, days: Optional[int] = None) -> BacktestResult:
    """
    Run backtest for a single symbol.
    
    Fetches historical data, scans for Trident patterns, simulates entries/exits.
    """
    if days is None:
        days = config.BACKTEST_DAYS

    result = BacktestResult(symbol=symbol)
    pip_value = get_pip_value(symbol)

    log.info(f"{'='*60}")
    log.info(f"Backtesting {symbol} | Last {days} days | Pip: {pip_value}")
    log.info(f"{'='*60}")

    # Fetch historical data
    date_to = datetime.now()
    date_from = date_to - timedelta(days=days)

    df_30m = mt5c.get_candles_range(symbol, config.ENTRY_TIMEFRAME, date_from, date_to)
    df_daily = mt5c.get_candles_range(symbol, config.BIAS_TIMEFRAME, date_from, date_to)

    if df_30m.empty:
        log.warning(f"No 30M data for {symbol}")
        return result

    log.info(f"Loaded {len(df_30m)} bars (30M) and {len(df_daily)} bars (Daily)")

    # Calculate EMAs on full dataset
    all_periods = config.EMA_FAST_PERIODS + [config.EMA_TREND_PERIOD]
    df_30m = calculate_emas(df_30m, all_periods)

    if not df_daily.empty:
        df_daily = calculate_emas(df_daily, config.EMA_FAST_PERIODS)

    # Sliding window scan — step by window_size to avoid duplicate detections
    window_size = 50  # Look at 50 candles at a time
    seen_entry_times = set()  # Track unique trades by entry time
    last_trade_idx = -window_size

    for start_idx in range(0, len(df_30m) - window_size, window_size // 2):
        # Skip if too close to last trade
        if start_idx - last_trade_idx < 10:
            continue

        window = df_30m.iloc[start_idx:start_idx + window_size].copy().reset_index(drop=True)

        # Scan for Trident Pattern (time check enabled for kill zone)
        signal = validate_trident_pattern(window, symbol, check_time=True)

        if signal:
            # Deduplicate: skip if we already found a trade at this entry time
            entry_key = (str(signal.signal_time), signal.entry_price, signal.direction)
            if entry_key in seen_entry_times:
                continue
            seen_entry_times.add(entry_key)

            # Map confirmation index back to main DataFrame
            actual_confirm_idx = start_idx + signal.confirmation_idx

            trade = BacktestTrade(
                symbol=symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                entry_time=signal.signal_time,
                stop_loss=signal.stop_loss,
            )

            # Simulate the trade forward
            trade = simulate_trade_exit(
                df_30m, df_daily, trade,
                actual_confirm_idx, pip_value
            )

            # Calculate R:R
            risk = abs(trade.entry_price - trade.stop_loss)
            if risk > 0:
                trade.rr_ratio = trade.pnl_pips * pip_value / risk
            else:
                trade.rr_ratio = 0.0

            result.trades.append(trade)
            last_trade_idx = start_idx

            win_mark = "[WIN]" if trade.result == "WIN" else "[LOSS]"
            log.info(f"  {win_mark} {trade.direction} @ {trade.entry_price:.5f} -> "
                     f"{trade.exit_price:.5f} | {trade.pnl_pips:+.1f} pips | "
                     f"R:R {trade.rr_ratio:+.1f} | {trade.exit_reason} | "
                     f"{trade.entry_time}")

    # Calculate summary stats
    result.total_trades = len(result.trades)
    result.wins = sum(1 for t in result.trades if t.result == "WIN")
    result.losses = sum(1 for t in result.trades if t.result == "LOSS")
    result.win_rate = (result.wins / result.total_trades * 100) if result.total_trades > 0 else 0
    result.total_pnl_pips = sum(t.pnl_pips for t in result.trades)

    rr_values = [t.rr_ratio for t in result.trades if t.rr_ratio != 0]
    result.avg_rr = np.mean(rr_values) if rr_values else 0.0

    pnl_values = [t.pnl_pips for t in result.trades]
    result.best_trade_pips = max(pnl_values) if pnl_values else 0.0
    result.worst_trade_pips = min(pnl_values) if pnl_values else 0.0

    # Max drawdown
    if pnl_values:
        cumulative = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        result.max_drawdown_pips = np.max(drawdowns)

    return result


def print_results(results: List[BacktestResult]):
    """Print formatted backtest results."""
    print("\n" + "=" * 80)
    print("                    BACKTEST RESULTS -- TRIDENT PATTERN")
    print("=" * 80)

    for r in results:
        print(f"\n{'-'*60}")
        print(f"  [SYMBOL] {r.symbol}")
        print(f"{'-'*60}")
        print(f"  Total Trades     : {r.total_trades}")
        print(f"  Wins / Losses    : {r.wins} / {r.losses}")
        print(f"  Win Rate         : {r.win_rate:.1f}%")
        print(f"  Total PnL (pips) : {r.total_pnl_pips:+.1f}")
        print(f"  Avg R:R          : {r.avg_rr:+.2f}")
        print(f"  Best Trade       : {r.best_trade_pips:+.1f} pips")
        print(f"  Worst Trade      : {r.worst_trade_pips:+.1f} pips")
        print(f"  Max Drawdown     : {r.max_drawdown_pips:.1f} pips")

    # Overall summary
    total_trades = sum(r.total_trades for r in results)
    total_wins = sum(r.wins for r in results)
    total_pnl = sum(r.total_pnl_pips for r in results)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"\n{'='*60}")
    print(f"  >> OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Trades     : {total_trades}")
    print(f"  Overall Win Rate : {overall_wr:.1f}%")
    print(f"  Total PnL (pips) : {total_pnl:+.1f}")
    print(f"{'='*60}\n")


def save_results_csv(results: List[BacktestResult], filename: str = "backtest_results.csv"):
    """Save all trades to a CSV file for further analysis."""
    rows = []
    for r in results:
        for t in r.trades:
            rows.append({
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "pnl_pips": t.pnl_pips,
                "rr_ratio": t.rr_ratio,
                "result": t.result,
                "exit_reason": t.exit_reason,
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        log.info(f"Results saved to {filename}")
    else:
        log.info("No trades to save.")


def main():
    parser = argparse.ArgumentParser(description="Backtest the Trident Pattern Strategy")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol to backtest (default: all configured)")
    parser.add_argument("--days", type=int, default=config.BACKTEST_DAYS,
                        help=f"Number of days to backtest (default: {config.BACKTEST_DAYS})")
    parser.add_argument("--output", type=str, default="backtest_results.csv",
                        help="CSV output filename")
    args = parser.parse_args()

    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    print("\n" + "=" * 60)
    print("  TG Capital Playbook -- Trident Pattern Backtest")
    print("=" * 60)

    # Connect to MT5 for historical data
    if not mt5c.connect():
        log.error("Failed to connect to MT5. Make sure MT5 is open and logged in.")
        sys.exit(1)

    try:
        symbols = [args.symbol] if args.symbol else config.SYMBOLS
        results = []

        for symbol in symbols:
            result = backtest_symbol(symbol, days=args.days)
            results.append(result)

        print_results(results)
        save_results_csv(results, args.output)

    finally:
        mt5c.disconnect()


if __name__ == "__main__":
    main()
