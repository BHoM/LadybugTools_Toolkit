import inspect
import json
import sys
import traceback
import uuid
from functools import wraps
from typing import Callable

from .encoder import BHoMEncoder
from .log import _bhom_version, _console_logger, _file_logger, _ticks

BHOM_VERSION = _bhom_version()  # defined here to save calling method multiple times

FILE_LOGGER = _file_logger()
CONSOLE_LOGGER = _console_logger()


def bhom_analytics(func: Callable) -> Callable:
    """The BHoM decorator used to capture usage analytics for called methods/functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """_"""
        _id = uuid.uuid4()

        d = {
            "BHoMVersion": BHOM_VERSION,
            "BHoM_Guid": _id,
            "CallerName": inspect.getmodule(func).__name__,
            "ComponentId": _id,
            "CustomData": {},
            "Errors": [],
            "FileId": "",
            "FileName": "",
            "Fragments": [],
            "Name": "",
            "ProjectID": "",
            "SelectedItem": {
                "Name": func.__name__,
                "_bhomVersion": BHOM_VERSION,
                "_t": "Python",
            },
            "Tags": [],
            "Time": {
                "$date": _ticks(),
            },
            "UI": "Python",
            "UiVersion": sys.version,
            "_bhomVersion": BHOM_VERSION,
            "_t": "BH.oM.UI.UsageLogEntry",
        }

        try:
            result = func(*args, **kwargs)
            FILE_LOGGER.info(json.dumps(d, sort_keys=True, cls=BHoMEncoder))
            # CONSOLE_LOGGER.info(json.dumps(d, sort_keys=True, cls=BHoMEncoder))
        except Exception as exc:
            d["Errors"].append(traceback.format_exc())
            FILE_LOGGER.info(json.dumps(d, sort_keys=True, cls=BHoMEncoder))
            # CONSOLE_LOGGER.info(json.dumps(d, sort_keys=True, cls=BHoMEncoder))
            raise exc

        return result

    return wrapper
