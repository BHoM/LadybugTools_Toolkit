from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import FunctionType, MethodType
from typing import Any, Dict

import numpy as np
from caseconverter import pascalcase, snakecase

from .analytics import bhom_analytics
from .encoder import BHoMEncoder


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


def keys_to_pascalcase(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all keys in a dictionary into pascalcase."""
    return {
        pascalcase(k)
        if k != "_t"
        else k: keys_to_pascalcase(v)
        if isinstance(v, dict)
        else v
        for k, v in dictionary.items()
    }


def keys_to_snakecase(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all keys in a dictionary into snakecase."""
    return {
        snakecase(k)
        if k != "_t"
        else k: keys_to_snakecase(v)
        if isinstance(v, dict)
        else v
        for k, v in dictionary.items()
    }


def inf_dtype_to_inf_str(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any values in a dict that are "Infinite" or "-Infinite"
    into a string equivalent."""

    def to_inf_str(v: Any):
        if isinstance(v, dict):
            return inf_dtype_to_inf_str(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return [to_inf_str(item) for item in v]
        if v in (math.inf, -math.inf):
            return json.dumps(v)
        return v

    return {k: to_inf_str(v) for k, v in dictionary.items()}


def inf_str_to_inf_dtype(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any values in a dict that are "inf" or "-inf" into a numeric
    equivalent."""
    if not isinstance(dictionary, dict):
        return dictionary

    def to_inf_dtype(v: Any):
        if isinstance(v, dict):
            return inf_str_to_inf_dtype(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return [to_inf_dtype(item) for item in v]
        if v in ("inf", "-inf"):
            return json.loads(v)
        return v

    return {k: to_inf_dtype(v) for k, v in dictionary.items()}


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

    return keys_to_snakecase(inf_str_to_inf_dtype(dictionary))
