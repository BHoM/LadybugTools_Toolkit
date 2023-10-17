"""Root for the bhom subpackage."""

from .analytics import decorator_factory
from .log import console_logger

CONSOLE_LOGGER = console_logger()
