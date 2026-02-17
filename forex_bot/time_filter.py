"""
Time Filter — validates London Kill Zone and FVG formation windows.
All times are converted to New York timezone.
"""

from datetime import datetime, time
import pytz
import config


NY_TZ = pytz.timezone(config.NY_TIMEZONE)

KILL_ZONE_START = time(config.KILL_ZONE_START_HOUR, config.KILL_ZONE_START_MINUTE)
KILL_ZONE_END   = time(config.KILL_ZONE_END_HOUR, config.KILL_ZONE_END_MINUTE)

FVG_WINDOW_START = time(config.FVG_WINDOW_START_HOUR, config.FVG_WINDOW_START_MINUTE)
FVG_WINDOW_END   = time(config.FVG_WINDOW_END_HOUR, config.FVG_WINDOW_END_MINUTE)


def to_ny_time(dt: datetime) -> datetime:
    """Convert a datetime to New York timezone. Handles naive and aware datetimes."""
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = pytz.utc.localize(dt)
    return dt.astimezone(NY_TZ)


def is_in_kill_zone(dt: datetime) -> bool:
    """Check if datetime falls within London Kill Zone (3:00–6:30 AM NY)."""
    ny_dt = to_ny_time(dt)
    current_time = ny_dt.time()
    return KILL_ZONE_START <= current_time <= KILL_ZONE_END


def is_in_fvg_window(dt: datetime) -> bool:
    """Check if datetime falls within FVG formation window (2:30–4:00 AM NY)."""
    ny_dt = to_ny_time(dt)
    current_time = ny_dt.time()
    return FVG_WINDOW_START <= current_time <= FVG_WINDOW_END


def is_weekday(dt: datetime) -> bool:
    """Check if it's a trading day (Monday–Friday)."""
    return dt.weekday() < 5  # 0=Mon, 4=Fri


def get_ny_now() -> datetime:
    """Get the current time in New York timezone."""
    return datetime.now(NY_TZ)
