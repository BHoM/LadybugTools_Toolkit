"""Logging utilities for BHoM analytics."""

# pylint: disable=E0401
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# pylint: enable=E0401

from .util import log_folder, toolkit_name

TOOLKIT_NAME = toolkit_name()


# console logger for progress reporting
def console_logger() -> logging.Logger:
    """Create/return the console logger."""
    logger = logging.getLogger(f"{TOOLKIT_NAME}[console]")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def analytics_logger() -> logging.Logger:
    """Create/return a logger, for outputting BHoM analytics to a file."""
    logger = logging.getLogger(TOOLKIT_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = RotatingFileHandler(
        log_folder() / f"{TOOLKIT_NAME}_{datetime.now().strftime('%Y%m%d')}.log",
        encoding="utf-8",
        mode="a",
        delay=True,  # wait until all logs collected before writing
        maxBytes=25 * 1024 * 1024,  # 25mb max before file overwritten
        backupCount=0,
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger
