import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import numpy as np
import pandas as pd
from typing import List
from ladybug.datacollection import HourlyContinuousCollection
from ladybug_extension.datacollection.from_series import from_series
from ladybug_extension.datacollection.to_series import to_series


def radiant_temperature_from_collections(
    collections: List[HourlyContinuousCollection], view_factors: List[float]
) -> HourlyContinuousCollection:
    """Calculate the radiant temperature from a list of hourly continuous collections and view factors to each of those collections.

    Args:
        collections (List[HourlyContinuousCollection]): A list of hourly continuous collections.
        view_factors (List[float]): A list of view factors to each of the collections.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of the effective radiant temperature.
    """

    if len(collections) != len(view_factors):
        raise ValueError("The number of collections and view factors must be the same.")
    if sum(view_factors) != 1:
        raise ValueError("The sum of view factors must be 1.")

    mrt_series = (
        np.power(
            (
                np.power(
                    pd.concat([to_series(i) for i in collections], axis=1) + 273.15, 4
                )
                * view_factors
            ).sum(axis=1),
            0.25,
        )
        - 273.15
    )
    mrt_series.name = "Temperature (C)"
    return from_series(mrt_series)
