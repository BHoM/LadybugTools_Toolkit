"""Methods for encoding and decoding objects from this toolkit into serialisable JSON."""

# pylint disable=too-few-public-methods
# pylint disable=too-many-return-statements

import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from caseconverter import pascalcase, snakecase
from ladybug.analysisperiod import AnalysisPeriod, DateTime
from ladybug.datacollection import BaseCollection
from ladybug.epw import EPW
from ladybug.wea import Wea
from ladybug_geometry.geometry2d import (
    Arc2D,
    LineSegment2D,
    Mesh2D,
    Point2D,
    Polygon2D,
    Polyline2D,
    Ray2D,
    Vector2D,
)
from ladybug_geometry.geometry3d import (
    Arc3D,
    Cone,
    Cylinder,
    Face3D,
    LineSegment3D,
    Mesh3D,
    Plane,
    Point3D,
    Polyface3D,
    Polyline3D,
    Ray3D,
    Sphere,
    Vector3D,
)

# TODO - Add support for encoding/decoding the imported objects. Unfinished as
# encoding currently handled within objects themselves to save development time.


def keys_to_pascalcase(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Convert all keys in a dictionary into pascalcase."""
    return {
        pascalcase(k)
        if (k != "_t") and isinstance(k, str)
        else k: keys_to_pascalcase(v)
        if isinstance(v, dict)
        else v
        for k, v in dictionary.items()
    }


def keys_to_snakecase(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Convert all keys in a dictionary into snakecase."""
    return {
        snakecase(k)
        if (k != "_t") and isinstance(k, str)
        else k: keys_to_snakecase(v)
        if isinstance(v, dict)
        else v
        for k, v in dictionary.items()
    }
