"""
Indicators — EMA calculations, EMA stacking, trend bias, and doji detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import config


def calculate_emas(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Add EMA columns to the DataFrame for each period.
    Columns are named 'ema_5', 'ema_9', etc.
    """
    if periods is None:
        periods = config.EMA_FAST_PERIODS + [config.EMA_TREND_PERIOD]

    for period in periods:
        col_name = f"ema_{period}"
        df[col_name] = df["close"].ewm(span=period, adjust=False).mean()

    return df


def are_emas_stacked(df: pd.DataFrame, direction: str, index: int = -1) -> bool:
    """
    Check if EMAs (5, 9, 13, 21) are cleanly stacked at a given candle index.

    For 'long':  ema_5 > ema_9 > ema_13 > ema_21
    For 'short': ema_5 < ema_9 < ema_13 < ema_21

    If any EMAs are crossing/tangled, returns False (setup invalid per strategy).
    """
    periods = config.EMA_FAST_PERIODS
    try:
        row = df.iloc[index]
    except IndexError:
        return False

    ema_values = [row[f"ema_{p}"] for p in periods]

    if direction == "long":
        # Each faster EMA must be above the slower one
        for i in range(len(ema_values) - 1):
            if ema_values[i] <= ema_values[i + 1]:
                return False
        return True
    elif direction == "short":
        # Each faster EMA must be below the slower one
        for i in range(len(ema_values) - 1):
            if ema_values[i] >= ema_values[i + 1]:
                return False
        return True

    return False


def get_200ema_bias(df: pd.DataFrame, index: int = -1) -> str:
    """
    Determine directional bias from 200 EMA.
    Returns 'long' if price is above 200 EMA, 'short' if below.
    """
    try:
        row = df.iloc[index]
    except IndexError:
        return "neutral"

    ema_col = f"ema_{config.EMA_TREND_PERIOD}"
    if ema_col not in df.columns:
        return "neutral"

    if row["close"] > row[ema_col]:
        return "long"
    elif row["close"] < row[ema_col]:
        return "short"
    return "neutral"


def is_doji(candle: pd.Series, threshold: Optional[float] = None) -> bool:
    """
    Check if a candle is a doji (small body relative to total range).
    A doji means indecision — body is ≤ threshold% of the high-low range.
    """
    if threshold is None:
        threshold = config.DOJI_BODY_RATIO

    body = abs(candle["close"] - candle["open"])
    total_range = candle["high"] - candle["low"]

    if total_range == 0:
        return False  # Zero-range candle is not a valid doji

    return (body / total_range) <= threshold


def candle_direction(candle: pd.Series) -> str:
    """Return 'bullish', 'bearish', or 'neutral' based on candle close vs open."""
    if candle["close"] > candle["open"]:
        return "bullish"
    elif candle["close"] < candle["open"]:
        return "bearish"
    return "neutral"
