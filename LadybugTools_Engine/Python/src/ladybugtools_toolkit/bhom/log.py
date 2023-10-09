"""Logging utilities for BHoM analytics."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from .util import log_folder, toolkit_name

TOOLKIT_NAME = toolkit_name()
BHOM_ANALYTICS_FORMATTER = logging.Formatter("%(message)s")


def get_logger() -> logging.Logger:
    """Create/return a logger, for outputting BHoM analytics to a file."""
    logger = logging.getLogger(TOOLKIT_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # TODO - enable for production
    # file_handler = RotatingFileHandler(
    #     log_folder() / f"{TOOLKIT_NAME}_{datetime.now().strftime('%Y%m%d')}.log",
    #     encoding="utf-8",
    #     mode="a",
    #     delay=True,  # wait until all logs collected before writing
    #     maxBytes=25 * 1024 * 1024,  # 25mb max before file overwritten
    #     backupCount=0,
    # )
    # file_handler.setFormatter(BHOM_ANALYTICS_FORMATTER)
    # file_handler.setLevel(logging.DEBUG)
    # logger.addHandler(file_handler)

    # TODO - disable for production
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(BHOM_ANALYTICS_FORMATTER)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    return logger
