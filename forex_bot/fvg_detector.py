"""
Fair Value Gap Detector — identifies 3-candle FVGs on the 30M chart.

A Bullish FVG:  candle1.high < candle3.low  → gap between candle1 high and candle3 low
A Bearish FVG:  candle1.low  > candle3.high → gap between candle1 low and candle3 high
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from time_filter import is_in_fvg_window


@dataclass
class FVG:
    """Represents a detected Fair Value Gap."""
    direction: str          # "bullish" or "bearish"
    top: float              # Upper boundary of the gap
    bottom: float           # Lower boundary of the gap
    midpoint: float         # 50% level (consequent encroachment)
    candle1_idx: int        # Index of the first candle in the FVG
    candle2_idx: int        # Index of the middle candle (impulse)
    candle3_idx: int        # Index of the third candle
    candle1_time: pd.Timestamp
    fvg_candle_low: float   # Low of the FVG-forming candle (for stop loss)
    fvg_candle_high: float  # High of the FVG-forming candle (for stop loss)


def find_fvgs(df: pd.DataFrame, check_time: bool = True) -> List[FVG]:
    """
    Scan the DataFrame for all Fair Value Gaps.
    
    A bullish FVG exists when candle3.low > candle1.high (gap up).
    A bearish FVG exists when candle3.high < candle1.low (gap down).
    
    Args:
        df: DataFrame with OHLC data
        check_time: If True, only consider FVGs formed in the FVG time window
    
    Returns:
        List of FVG objects found
    """
    fvgs = []

    if len(df) < 3:
        return fvgs

    for i in range(len(df) - 2):
        candle1 = df.iloc[i]
        candle2 = df.iloc[i + 1]  # The impulse candle
        candle3 = df.iloc[i + 2]

        # Check time window if required
        if check_time and "time" in df.columns:
            if not is_in_fvg_window(candle2["time"]):
                continue

        # Bullish FVG: gap between candle1 high and candle3 low
        if candle3["low"] > candle1["high"]:
            top = candle3["low"]
            bottom = candle1["high"]
            midpoint = (top + bottom) / 2.0

            fvg = FVG(
                direction="bullish",
                top=top,
                bottom=bottom,
                midpoint=midpoint,
                candle1_idx=i,
                candle2_idx=i + 1,
                candle3_idx=i + 2,
                candle1_time=candle1.get("time", pd.NaT),
                fvg_candle_low=candle2["low"],     # Impulse candle low for SL
                fvg_candle_high=candle2["high"],
            )
            fvgs.append(fvg)

        # Bearish FVG: gap between candle1 low and candle3 high
        elif candle3["high"] < candle1["low"]:
            top = candle1["low"]
            bottom = candle3["high"]
            midpoint = (top + bottom) / 2.0

            fvg = FVG(
                direction="bearish",
                top=top,
                bottom=bottom,
                midpoint=midpoint,
                candle1_idx=i,
                candle2_idx=i + 1,
                candle3_idx=i + 2,
                candle1_time=candle1.get("time", pd.NaT),
                fvg_candle_low=candle2["low"],
                fvg_candle_high=candle2["high"],   # Impulse candle high for SL
            )
            fvgs.append(fvg)

    return fvgs


def find_latest_fvg(df: pd.DataFrame, direction: str,
                    check_time: bool = True) -> Optional[FVG]:
    """
    Find the most recent FVG matching the given direction.
    Returns None if no FVG is found.
    """
    fvgs = find_fvgs(df, check_time=check_time)
    matching = [f for f in fvgs if f.direction == direction]

    if not matching:
        return None

    return matching[-1]  # Most recent


def get_fvg_midpoint(fvg: FVG) -> float:
    """Return the 50% level (consequent encroachment) of an FVG."""
    return fvg.midpoint
