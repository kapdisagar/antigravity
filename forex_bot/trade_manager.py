"""
Trade Manager — handles order execution, stop loss, take profit,
and daily chart exit monitoring.
"""

import pandas as pd
from typing import Optional
import config
from logger import setup_logger, log_trade
from trident_pattern import TradeSignal

log = setup_logger()


def execute_entry(signal: TradeSignal, mt5_conn) -> Optional[dict]:
    """
    Execute a trade entry based on a Trident Pattern signal.
    Uses the MT5 connector to place the order.
    
    Args:
        signal: The validated TradeSignal
        mt5_conn: The mt5_connector module
    
    Returns:
        Order result dict, or None on failure
    """
    # Check max open trades
    open_positions = mt5_conn.get_open_positions()
    bot_positions = [p for p in open_positions if p["symbol"] == signal.symbol]

    if len(open_positions) >= config.MAX_OPEN_TRADES:
        log.warning(f"Max open trades ({config.MAX_OPEN_TRADES}) reached. Skipping {signal.symbol}.")
        return None

    # Determine stop loss
    sl = signal.stop_loss if signal.use_hard_sl else 0.0

    # Place the order
    result = mt5_conn.place_order(
        symbol=signal.symbol,
        order_type=signal.direction,
        lot=config.LOT_SIZE,
        sl=sl,
        tp=0.0,  # TP managed via daily chart monitoring
        comment=f"Trident_{signal.direction}"
    )

    if result is not None:
        log_trade({
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "stop_loss": sl,
            "take_profit": "Daily chart managed",
            "lot_size": config.LOT_SIZE,
            "result": "OPEN",
            "notes": f"FVG midpoint: {signal.fvg.midpoint:.5f}"
        })

    return result


def should_exit_on_daily(daily_df: pd.DataFrame, direction: str) -> bool:
    """
    Check the daily chart for exit conditions:
    1. EMAs begin to reverse direction
    2. A significant opposing candlestick appears
    
    Args:
        daily_df: Daily OHLC DataFrame with EMAs calculated
        direction: "BUY" or "SELL"
    
    Returns:
        True if the position should be closed
    """
    from indicators import are_emas_stacked

    if len(daily_df) < 3:
        return False

    last_candle = daily_df.iloc[-1]
    prev_candle = daily_df.iloc[-2]

    if direction == "BUY":
        # Exit if EMAs are no longer stacked bullish
        emas_ok = are_emas_stacked(daily_df, "long", -1)
        if not emas_ok:
            # Check if they were stacked on the previous candle
            prev_stacked = are_emas_stacked(daily_df, "long", -2)
            if prev_stacked:
                log.info("📊 Daily EMAs breaking bullish stack — exit signal")
                return True

        # Exit on significant bearish candle
        body = abs(last_candle["close"] - last_candle["open"])
        total_range = last_candle["high"] - last_candle["low"]
        if total_range > 0 and body / total_range > 0.7:
            if last_candle["close"] < last_candle["open"]:
                # Strong bearish candle
                # Also check if it closed below previous candle's low
                if last_candle["close"] < prev_candle["low"]:
                    log.info("📊 Significant bearish daily candle — exit signal")
                    return True

    elif direction == "SELL":
        # Exit if EMAs are no longer stacked bearish
        emas_ok = are_emas_stacked(daily_df, "short", -1)
        if not emas_ok:
            prev_stacked = are_emas_stacked(daily_df, "short", -2)
            if prev_stacked:
                log.info("📊 Daily EMAs breaking bearish stack — exit signal")
                return True

        # Exit on significant bullish candle
        body = abs(last_candle["close"] - last_candle["open"])
        total_range = last_candle["high"] - last_candle["low"]
        if total_range > 0 and body / total_range > 0.7:
            if last_candle["close"] > last_candle["open"]:
                if last_candle["close"] > prev_candle["high"]:
                    log.info("📊 Significant bullish daily candle — exit signal")
                    return True

    return False


def check_gold_candle_filter(df_30m: pd.DataFrame, entry_price: float,
                              direction: str) -> bool:
    """
    Gold-specific exit: instead of hard SL, monitor candle closes.
    If a 30M candle CLOSES beyond the stop level, signal exit.
    
    This prevents getting stopped out by deep liquidity wicks that
    Gold is known for.
    
    Returns True if the position should be closed.
    """
    if len(df_30m) < 1:
        return False

    last_candle = df_30m.iloc[-1]

    if direction == "BUY":
        # If candle CLOSES below entry by a significant amount, exit
        # We use a generous filter since Gold makes deep wicks
        if last_candle["close"] < entry_price * 0.995:  # 0.5% below entry
            log.info("🥇 Gold candle close filter triggered — exit LONG")
            return True
    elif direction == "SELL":
        if last_candle["close"] > entry_price * 1.005:
            log.info("🥇 Gold candle close filter triggered — exit SHORT")
            return True

    return False
