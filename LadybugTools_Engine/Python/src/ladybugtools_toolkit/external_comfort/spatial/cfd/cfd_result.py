from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.cfd.cfd_directory import (
    cfd_directory,
)


class CFDResult:
    """An object containing results from a CFD simulation.
    # These results should be extracted from a number of CFD simulations at the same point locations
    # as in the SpatialComfort simulation, for upwards of 8 (cardinal) directions. Results should be saved in the
    """

    def __init__(self) -> CFDResult:
        self.i = None

    @classmethod
    def from_directory(cls, cfd_directory: Path) -> CFDResult:
        return None

    @staticmethod
    def _unique_wind_speed_direction(epw: EPW) -> List[List[float]]:
        """Create a list of unique Wind Speed and Wind Direction values for a given EPW"""
        wind_speed_directions = np.stack(
            [epw.wind_speed.values, epw.wind_direction.values], axis=1
        )
        return np.unique(wind_speed_directions, axis=0)
