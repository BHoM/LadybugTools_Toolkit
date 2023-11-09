"""BHoM analytics decorator."""
# pylint: disable=E0401
import inspect
import json
import sys
import uuid
from functools import wraps
from typing import Any, Callable

# pylint: enable=E0401
from .log import ANALYTICS_LOGGER
from .util import bhom_version, csharp_ticks, toolkit_name

BHoM_VERSION = bhom_version()


def decorator_factory(disable: bool = False) -> Callable:
    """Decorator for capturing usage data.

    Arguments
    ---------
    disable : bool, optional
        Whether to disable the decorator, by default False

    Returns
    -------
    Callable
        The decorated function.
    """

    def decorator(function: Callable):
        """A decorator to capture usage data for called methods/functions.

        Arguments
        ---------
        function : Callable
            The function to decorate.

        Returns
        -------
        Callable
            The decorated function.
        """

        @wraps(function)
        def wrapper(*args, **kwargs) -> Any:
            """A wrapper around the function that captures usage analytics."""

            if disable:
                return function(*args, **kwargs)

            _id = uuid.uuid4()

            # get the data being passed to the function, expected dtype and return type
            argspec = inspect.getfullargspec(function)[-1]
            argspec.pop("return", None)

            _args = [f'{{"_t": "{argspec[k]}", "Name": "{k}"}}' for k in argspec.keys()]

            exec_metadata = {
                "BHoMVersion": bhom_version(),
                "BHoM_Guid": _id,
                "CallerName": function.__name__,
                "ComponentId": _id,
                "CustomData": {},
                "Errors": [],
                "FileId": "",
                "FileName": "",
                "Fragments": [],
                "Name": "",
                # TODO - get project properties from another function/logging
                # method (or from BHoM DLL analytics capture ...)
                "ProjectID": "",
                "SelectedItem": {
                    "MethodName": function.__name__,
                    "Parameters": _args,
                    "TypeName": f"{function.__module__}.{function.__qualname__}",
                    "_bhomVersion": BHoM_VERSION,
                    "_t": "Python",
                },
                "Time": {
                    "$date": csharp_ticks(short=True),
                },
                "UI": "Python",
                "UiVersion": toolkit_name(),
                "_t": "BH.oM.UI.UsageLogEntry",
            }

            try:
                result = function(*args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                exec_metadata["Errors"].extend(sys.exc_info())
                raise exc
            finally:
                ANALYTICS_LOGGER.info(
                    json.dumps(exec_metadata, default=str, indent=None)
                )

            return result

        return wrapper

    return decorator
