from python_toolkit.bhom.logging import CONSOLE_LOGGER as PYTHON_TOOLKIT_LOGGER
import logging

from .. import TOOLKIT_NAME

CONSOLE_LOGGER = logging.getLogger(f"{TOOLKIT_NAME}[console]")
CONSOLE_LOGGER.parent = PYTHON_TOOLKIT_LOGGER
CONSOLE_LOGGER.propagate = True
CONSOLE_LOGGER.setLevel(logging.DEBUG)
#logger handler is handled by the parent logger