"""
MT5 Connector — handles MetaTrader 5 connection, data retrieval, and order execution.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import config
from logger import setup_logger

log = setup_logger()


def connect():
    """Initialize connection to the MT5 terminal."""
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False

    account_info = mt5.account_info()
    if account_info is None:
        log.error("Failed to get account info — is MT5 logged in?")
        mt5.shutdown()
        return False

    log.info(f"Connected to MT5 | Account: {account_info.login} | "
             f"Server: {account_info.server} | "
             f"Balance: {account_info.balance:.2f} {account_info.currency}")
    return True


def disconnect():
    """Shutdown MT5 connection."""
    mt5.shutdown()
    log.info("MT5 connection closed.")


def get_timeframe_constant(tf_name: str):
    """Convert config timeframe string to MT5 constant."""
    tf_map = {
        "TIMEFRAME_M1":  mt5.TIMEFRAME_M1,
        "TIMEFRAME_M5":  mt5.TIMEFRAME_M5,
        "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
        "TIMEFRAME_M30": mt5.TIMEFRAME_M30,
        "TIMEFRAME_H1":  mt5.TIMEFRAME_H1,
        "TIMEFRAME_H4":  mt5.TIMEFRAME_H4,
        "TIMEFRAME_D1":  mt5.TIMEFRAME_D1,
        "TIMEFRAME_W1":  mt5.TIMEFRAME_W1,
    }
    return tf_map.get(tf_name, mt5.TIMEFRAME_M30)


def get_candles(symbol: str, timeframe_name: str, count: int = 200) -> pd.DataFrame:
    """Fetch OHLC candles as a DataFrame. Returns empty DataFrame on failure."""
    tf = get_timeframe_constant(timeframe_name)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None or len(rates) == 0:
        log.warning(f"No candle data for {symbol} on {timeframe_name}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={
        "time": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume"
    }, inplace=True)
    return df


def get_candles_range(symbol: str, timeframe_name: str,
                      date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """Fetch candles within a specific date range. Used mainly for backtesting."""
    tf = get_timeframe_constant(timeframe_name)
    rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)

    if rates is None or len(rates) == 0:
        log.warning(f"No candle data for {symbol} between {date_from} and {date_to}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def get_account_info() -> dict:
    """Return account balance, equity, margin, etc."""
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "login": info.login,
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "currency": info.currency,
        "leverage": info.leverage,
    }


def get_symbol_info(symbol: str) -> dict:
    """Get symbol point size, digits, and other info for order calculations."""
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"Symbol {symbol} not found in MT5")
        return {}
    return {
        "point": info.point,
        "digits": info.digits,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
    }


def place_order(symbol: str, order_type: str, lot: float,
                sl: float = 0.0, tp: float = 0.0, comment: str = "TridentBot"):
    """
    Place a market order.
    order_type: "BUY" or "SELL"
    Returns the order result or None on failure.
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        log.error(f"Symbol {symbol} not found")
        return None

    # Ensure symbol is visible in Market Watch
    if not sym_info.visible:
        mt5.symbol_select(symbol, True)

    price = mt5.symbol_info_tick(symbol)
    if price is None:
        log.error(f"Failed to get tick for {symbol}")
        return None

    if order_type == "BUY":
        trade_type = mt5.ORDER_TYPE_BUY
        entry_price = price.ask
    else:
        trade_type = mt5.ORDER_TYPE_SELL
        entry_price = price.bid

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      trade_type,
        "price":     entry_price,
        "sl":        sl,
        "tp":        tp,
        "deviation": config.SLIPPAGE,
        "magic":     config.MAGIC_NUMBER,
        "comment":   comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"Order send returned None: {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed | {symbol} {order_type} | "
                  f"retcode={result.retcode} | comment={result.comment}")
        return None

    log.info(f"✅ Order executed | {symbol} {order_type} {lot} lots @ {entry_price:.5f} | "
             f"SL={sl:.5f} TP={tp:.5f} | ticket={result.order}")
    return result


def close_position(ticket: int):
    """Close an open position by ticket number."""
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        log.warning(f"No position found with ticket {ticket}")
        return None

    pos = positions[0]
    symbol = pos.symbol
    lot = pos.volume

    # Reverse the direction to close
    if pos.type == mt5.ORDER_TYPE_BUY:
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      trade_type,
        "position":  ticket,
        "price":     price,
        "deviation": config.SLIPPAGE,
        "magic":     config.MAGIC_NUMBER,
        "comment":   "TridentBot_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Failed to close ticket {ticket}: {mt5.last_error()}")
        return None

    log.info(f"🔒 Position closed | ticket={ticket} | {symbol} @ {price:.5f}")
    return result


def get_open_positions() -> list:
    """Return all open positions placed by this bot (filtered by magic number)."""
    positions = mt5.positions_get()
    if positions is None:
        return []

    bot_positions = [
        {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": p.volume,
            "open_price": p.price_open,
            "current_price": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "open_time": datetime.fromtimestamp(p.time),
            "magic": p.magic,
        }
        for p in positions
        if p.magic == config.MAGIC_NUMBER
    ]
    return bot_positions
