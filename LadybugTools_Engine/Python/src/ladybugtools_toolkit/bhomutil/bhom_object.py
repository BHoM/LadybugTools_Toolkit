from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import FunctionType, MethodType
from typing import Any, Dict

import numpy as np

from .analytics import bhom_analytics
from .encoder import (
    BHoMEncoder,
    inf_dtype_to_inf_str,
    inf_str_to_inf_dtype,
    keys_to_pascalcase,
    keys_to_snakecase,
)


def is_decorated(func):
    """Check if a function is decorated with a BHoM analytics decorator."""
    return hasattr(func, "__wrapped__") or func.__name__ not in globals()


@dataclass(init=True)
class BHoMObject:
    """The base object all objects within this toolkit should inherit from."""

    _t: str = field(init=False, repr=False, default="BH.oM.LadybugTools.BHoMObject")

    def __post_init__(self):
        """Replace each method in class with a decorated version of that method."""
        if "SpatialComfort" not in str(self):
            for key in dir(self):
                if key.startswith("_"):
                    continue
                if isinstance(getattr(self, key), (FunctionType, MethodType)):
                    if is_decorated(getattr(self, key)):
                        continue
                    setattr(self, key, bhom_analytics(getattr(self, key)))

    def to_dict(self) -> Dict[str, Any]:
        """Return the BHoM-flavoured dictionary representation of this object."""
        dictionary = {}
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), FunctionType):
                continue
            dictionary[k] = v

        dictionary = dict_to_bhom_dict(dictionary)
        dictionary["_t"] = self._t

        return dictionary

    def to_json(self) -> str:
        """Return the BHoM-flavoured JSON string representation of this object."""
        return json.dumps(self.to_dict(), cls=BHoMEncoder)


def dict_to_bhom_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a regular Python-flavoured dictionary into a BHoM-flavoured
    dictionary.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be BHoM-ified.

    Returns:
        Dict[str, Any]: The BHoM-flavoured dictionary.
    """

    return keys_to_pascalcase(inf_dtype_to_inf_str(dictionary))


def bhom_dict_to_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a BHoM-flavoured dictionary into a regular Python-flavoured
    dictionary.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be Python-ified.

    Returns:
        Dict[str, Any]: The Python-flavoured dictionary.
    """
    d = keys_to_snakecase(inf_str_to_inf_dtype(dictionary))

    # check here to replace "x": { "_t": "BH.XYZ", "_v": [0, ..., n] } with "x": [0, ..., n]
    #
    return d
