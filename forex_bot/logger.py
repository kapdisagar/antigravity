"""
Logging module for the trading bot.
Logs to both console and file, with CSV trade logging.
"""

import logging
import csv
import os
from datetime import datetime
import config


def setup_logger(name="TridentBot"):
    """Set up a logger that writes to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-init
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    return logger


def log_trade(trade_data: dict):
    """Append a trade record to the CSV trade log."""
    file_exists = os.path.exists(config.LOG_TRADES_CSV)
    fieldnames = [
        "timestamp", "symbol", "direction", "entry_price",
        "stop_loss", "take_profit", "lot_size", "result",
        "pnl", "rr_ratio", "notes"
    ]

    with open(config.LOG_TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {k: trade_data.get(k, "") for k in fieldnames}
        if "timestamp" not in trade_data or not trade_data["timestamp"]:
            row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(row)
