"""Centralized logging configuration for the AI Interview Preparation Platform backend."""

import logging
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None, app_env: str = "development") -> None:
    """Configure root application logging with a clean structured format.

    Avoids duplicating handlers if logging has already been configured (e.g., under pytest).
    """
    resolved_level_name = (level or ("DEBUG" if app_env == "development" else "INFO")).upper()
    resolved_level = getattr(logging, resolved_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        root_logger.addHandler(handler)

    root_logger.setLevel(resolved_level)

    # Reduce noisy third-party client logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

