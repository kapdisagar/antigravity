"""
Trident Pattern Validator — the core pattern detection logic.

The Trident Pattern sequence:
1. A Fair Value Gap (FVG) forms on the 30M chart
2. A doji candle forms next, wicking into the FVG's 50% level
3. The confirmation candle closes below the doji's high (for longs)
   or above the doji's low (for shorts)
4. EMAs must be stacked in the trade direction
5. Price must be on the correct side of the 200 EMA
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from fvg_detector import FVG, find_fvgs
from indicators import are_emas_stacked, get_200ema_bias, is_doji
from time_filter import is_in_kill_zone


@dataclass
class TradeSignal:
    """Represents a validated Trident Pattern trade signal."""
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry_price: float      # Price at confirmation candle close
    stop_loss: float        # Below FVG candle low (longs) or above high (shorts)
    fvg: FVG                # The FVG that triggered the setup
    doji_idx: int           # Index of the doji candle
    confirmation_idx: int   # Index of the confirmation candle
    signal_time: pd.Timestamp
    use_hard_sl: bool       # Whether to use hard SL (False for Gold)


def validate_trident_pattern(df: pd.DataFrame, symbol: str,
                              check_time: bool = True) -> Optional[TradeSignal]:
    """
    Scan the candle data for a complete Trident Pattern.
    
    Process:
    1. Find FVGs in the data
    2. For each FVG, check if the next candle is a doji wicking into FVG 50%
    3. Check if the candle after the doji is a valid confirmation
    4. Verify EMA stacking and 200 EMA bias
    5. Verify time is within kill zone
    
    Returns a TradeSignal if a valid pattern is found, None otherwise.
    """
    import config

    if len(df) < 6:
        return None

    # Get all FVGs
    fvgs = find_fvgs(df, check_time=check_time)

    if not fvgs:
        return None

    # Check each FVG for the Trident Pattern (most recent first)
    for fvg in reversed(fvgs):
        doji_idx = fvg.candle3_idx + 1      # Candle right after FVG
        confirm_idx = fvg.candle3_idx + 2    # Candle after doji

        # Make sure we have enough candles
        if confirm_idx >= len(df):
            continue

        doji_candle = df.iloc[doji_idx]
        confirm_candle = df.iloc[confirm_idx]

        # ─── Step 1: Check doji ─────────────────────────────────────────
        if not is_doji(doji_candle):
            continue

        # ─── Step 2: Check doji wicks into FVG 50% ─────────────────────
        if fvg.direction == "bullish":
            # Doji must wick DOWN into the FVG midpoint zone
            # The low of the doji should reach near or below the FVG midpoint
            if doji_candle["low"] > fvg.midpoint:
                continue  # Doji didn't wick deep enough

            # ─── Step 3: Confirmation candle ────────────────────────────
            # Must close below doji's high (per strategy PDF)
            if confirm_candle["close"] >= doji_candle["high"]:
                # Actually for bullish, confirmation should show continuation
                # The PDF says "close below doji high" — meaning it doesn't
                # break above the doji, confirming a controlled pattern
                pass  # This is actually the VALID condition per strategy
            
            # Re-check: For a BULLISH trade, the confirmation candle should
            # close BELOW the doji's high (controlled move, not explosive)
            if confirm_candle["close"] > doji_candle["high"]:
                continue  # Invalid — broke above doji high

            # ─── Step 4: EMA stacking ──────────────────────────────────
            if not are_emas_stacked(df, "long", confirm_idx):
                continue

            # ─── Step 5: 200 EMA bias ──────────────────────────────────
            bias = get_200ema_bias(df, confirm_idx)
            if bias != "long":
                continue

            # ─── Step 6: Time check ────────────────────────────────────
            if check_time and "time" in df.columns:
                if not is_in_kill_zone(confirm_candle["time"]):
                    continue

            # ─── Valid bullish Trident Pattern! ────────────────────────
            is_gold = symbol.upper() in ["XAUUSD", "GOLD"]
            use_hard_sl = not is_gold or config.GOLD_USE_HARD_SL

            return TradeSignal(
                symbol=symbol,
                direction="BUY",
                entry_price=confirm_candle["close"],
                stop_loss=fvg.fvg_candle_low,  # Below the FVG impulse candle low
                fvg=fvg,
                doji_idx=doji_idx,
                confirmation_idx=confirm_idx,
                signal_time=confirm_candle.get("time", pd.NaT),
                use_hard_sl=use_hard_sl,
            )

        elif fvg.direction == "bearish":
            # Doji must wick UP into the FVG midpoint zone
            if doji_candle["high"] < fvg.midpoint:
                continue  # Doji didn't wick deep enough

            # Confirmation candle must close above doji's low (for shorts)
            if confirm_candle["close"] < doji_candle["low"]:
                continue  # Invalid for short

            # EMA stacking for shorts
            if not are_emas_stacked(df, "short", confirm_idx):
                continue

            # 200 EMA bias
            bias = get_200ema_bias(df, confirm_idx)
            if bias != "short":
                continue

            # Time check
            if check_time and "time" in df.columns:
                if not is_in_kill_zone(confirm_candle["time"]):
                    continue

            # Valid bearish Trident Pattern!
            is_gold = symbol.upper() in ["XAUUSD", "GOLD"]
            use_hard_sl = not is_gold or config.GOLD_USE_HARD_SL

            return TradeSignal(
                symbol=symbol,
                direction="SELL",
                entry_price=confirm_candle["close"],
                stop_loss=fvg.fvg_candle_high,  # Above FVG impulse candle high
                fvg=fvg,
                doji_idx=doji_idx,
                confirmation_idx=confirm_idx,
                signal_time=confirm_candle.get("time", pd.NaT),
                use_hard_sl=use_hard_sl,
            )

    return None


def scan_for_signals(df: pd.DataFrame, symbol: str,
                     check_time: bool = True) -> Optional[TradeSignal]:
    """
    High-level function to scan for Trident Pattern signals.
    This is the main entry point used by the bot loop and backtest.
    """
    return validate_trident_pattern(df, symbol, check_time=check_time)
