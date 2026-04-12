"""
Structured logging with correlation IDs per Telegram update.
Outputs to both console (stdout) and file (data/bot.log).
"""

import logging
import os
import sys
import contextvars
from typing import Optional
from logging.handlers import RotatingFileHandler

# Context variable for correlation ID (update_id)
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(update_id: int | str | None) -> None:
    _correlation_id.set(str(update_id) if update_id else None)


def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(correlation_id)s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with structured format — console + file."""
    root = logging.getLogger("bot")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not root.handlers:
        fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
        filt = CorrelationFilter()

        # Console handler (stdout) — always present
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        console.addFilter(filt)
        console.setLevel(logging.INFO)
        root.addHandler(console)

        # File handler (data/bot.log) — rotating, 5MB max, 3 backups
        try:
            log_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "bot.log")
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(fmt)
            file_handler.addFilter(filt)
            file_handler.setLevel(logging.DEBUG)
            root.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file: {e}", file=sys.stderr)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'bot' namespace."""
    return logging.getLogger(f"bot.{name}")
