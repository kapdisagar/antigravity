"""
TG Capital Playbook — Trident Pattern Trading Bot (Main Loop)

This bot runs during the London Kill Zone and scans for Trident Pattern
setups on configured forex/gold pairs. It connects to MetaTrader 5 for
live/demo execution.

Usage:
    python main.py
"""

import time
import sys
from datetime import datetime, timedelta

import config
import mt5_connector as mt5c
from indicators import calculate_emas
from trident_pattern import scan_for_signals
from trade_manager import execute_entry, should_exit_on_daily, check_gold_candle_filter
from time_filter import is_in_kill_zone, is_weekday, get_ny_now
from logger import setup_logger

log = setup_logger()

SCAN_INTERVAL = 30  # seconds between scans


def check_daily_limit():
    """
    Check if the daily loss limit or minimum balance limit has been hit.
    Returns True if trading is allowed, False if limits are breached.
    """
    account = mt5c.get_account_info()
    if not account:
        return False

    # 1. Check absolute balance limit
    if account['balance'] < config.MIN_BALANCE_LIMIT:
        log.warning(f"🛑 CRITICAL: Balance ({account['balance']}) is below MIN_BALANCE_LIMIT ({config.MIN_BALANCE_LIMIT}). Trading disabled.")
        return False

    # 2. Check Daily Loss (Realized PnL from today)
    # Note: This is an approximation based on account history
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta
    
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    
    # Get deals for today
    history = mt5.history_deals_get(today_start, now + timedelta(days=1))
    daily_pnl = 0.0
    if history:
        for deal in history:
            daily_pnl += deal.profit
            
    # Check floating PnL
    open_positions = mt5c.get_open_positions()
    floating_pnl = sum(p['profit'] for p in open_positions)
    
    total_today_pnl = daily_pnl + floating_pnl
    
    if total_today_pnl <= -config.DAILY_LOSS_LIMIT:
        log.warning(f"🛑 DAILY LIMIT HIT: Today's PnL ({total_today_pnl:.2f}) reached DAILY_LOSS_LIMIT (-{config.DAILY_LOSS_LIMIT}). Trading disabled for today.")
        return False
        
    return True


def print_banner():
    """Print startup banner."""
    banner = """
    ============================================================
    |     TG Capital Playbook -- Trident Pattern Bot           |
    |                                                          |
    |   Strategy : Trident Pattern (FVG + Doji + Confirm)      |
    |   Session  : London Kill Zone (3:00-6:30 AM NY)          |
    |   Pairs    : XAUUSD, USDJPY, EURUSD, GBPUSD, etc.       |
    ============================================================
    """
    print(banner)


def scan_symbols():
    """Scan all configured symbols for Trident Pattern setups."""
    signals_found = 0

    for symbol in config.SYMBOLS:
        try:
            # Fetch 30M candles for entry analysis
            df_30m = mt5c.get_candles(symbol, config.ENTRY_TIMEFRAME, count=200)
            if df_30m.empty:
                continue

            # Calculate EMAs
            all_periods = config.EMA_FAST_PERIODS + [config.EMA_TREND_PERIOD]
            df_30m = calculate_emas(df_30m, all_periods)

            # Scan for Trident Pattern
            signal = scan_for_signals(df_30m, symbol, check_time=True)

            if signal:
                log.info(f"🔔 TRIDENT SIGNAL | {symbol} {signal.direction} | "
                         f"Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f}")

                # Execute the trade
                result = execute_entry(signal, mt5c)
                if result:
                    signals_found += 1

        except Exception as e:
            log.error(f"Error scanning {symbol}: {e}")

    return signals_found


def monitor_open_positions():
    """Check open positions for exit conditions using daily chart."""
    positions = mt5c.get_open_positions()

    for pos in positions:
        symbol = pos["symbol"]
        direction = pos["type"]
        ticket = pos["ticket"]

        try:
            # Get daily chart data for exit analysis
            df_daily = mt5c.get_candles(symbol, config.BIAS_TIMEFRAME, count=100)
            if df_daily.empty:
                continue

            df_daily = calculate_emas(df_daily, config.EMA_FAST_PERIODS)

            # Check daily exit conditions
            if should_exit_on_daily(df_daily, direction):
                log.info(f"📤 Closing {symbol} {direction} (ticket {ticket}) — daily exit signal")
                mt5c.close_position(ticket)
                continue

            # Gold-specific candle close filter
            is_gold = symbol.upper() in ["XAUUSD", "GOLD"]
            if is_gold and not config.GOLD_USE_HARD_SL:
                df_30m = mt5c.get_candles(symbol, config.ENTRY_TIMEFRAME, count=10)
                if not df_30m.empty:
                    if check_gold_candle_filter(df_30m, pos["open_price"], direction):
                        log.info(f"🥇 Closing Gold {direction} (ticket {ticket}) — candle filter")
                        mt5c.close_position(ticket)

        except Exception as e:
            log.error(f"Error monitoring position {ticket}: {e}")


def main():
    """Main bot loop."""
    # Fix Windows console encoding
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    print_banner()
    log.info(f"Mode: {'DEMO' if config.DEMO_MODE else '⚠️  LIVE'}")
    log.info(f"Symbols: {', '.join(config.SYMBOLS)}")
    log.info(f"Lot Size: {config.LOT_SIZE} | Max Trades: {config.MAX_OPEN_TRADES}")

    # Connect to MT5
    if not mt5c.connect():
        log.error("Failed to connect to MT5. Exiting.")
        sys.exit(1)

    try:
        log.info("Bot started. Waiting for London Kill Zone...")
        while True:
            ny_now = get_ny_now()

            # Only trade on weekdays
            if not is_weekday(ny_now):
                log.debug("Weekend — market closed.")
                time.sleep(60)
                continue

            # Check if we're in the Kill Zone
            if is_in_kill_zone(ny_now):
                log.info(f"⏰ Inside Kill Zone | NY time: {ny_now.strftime('%H:%M:%S')}")
                
                # ENFORCE LOSS LIMITS
                if check_daily_limit():
                    scan_symbols()
                else:
                    log.info("⏳ Waiting for limits to reset or balance to increase...")
            else:
                log.debug(f"Outside Kill Zone | NY time: {ny_now.strftime('%H:%M:%S')}")

            # Always monitor open positions (exits can happen anytime)
            monitor_open_positions()

            # Show account status periodically
            account = mt5c.get_account_info()
            if account:
                log.debug(f"💰 Balance: {account['balance']:.2f} | "
                          f"Equity: {account['equity']:.2f} | "
                          f"Positions: {len(mt5c.get_open_positions())}")

            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log.info("\n🛑 Bot stopped by user.")
    finally:
        mt5c.disconnect()


if __name__ == "__main__":
    main()
