from typing import List

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection

from ...ladybug_extension.datacollection import from_series, to_series


def longwave_mean_radiant_temperature(
    collections: List[HourlyContinuousCollection], view_factors: List[float]
) -> HourlyContinuousCollection:
    """Calculate the LW MRT from a list of surface temperature collections, and view
        factors to each of those surfaces.

    Args:
        collections (List[HourlyContinuousCollection]): A list of hourly continuous collections.
        view_factors (List[float]): A list of view factors to each of the collections.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of the effective radiant
            temperature.
    """

    if len(collections) != len(view_factors):
        raise ValueError("The number of collections and view factors must be the same.")
    if sum(view_factors) != 1:
        raise ValueError("The sum of view factors must be 1.")

    mrt_series = (
        np.power(
            (
                np.power(
                    pd.concat([to_series(i) for i in collections], axis=1) + 273.15,
                    4,
                )
                * view_factors
            ).sum(axis=1),
            0.25,
        )
        - 273.15
    )
    mrt_series.name = "Mean Radiant Temperature (C)"
    return from_series(mrt_series)
