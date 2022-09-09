from typing import List

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from scipy.interpolate import interp1d


def unshaded_shaded_interpolation(
    unshaded: HourlyContinuousCollection,
    shaded: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view: pd.DataFrame,
    sun_up_bool: List[List[bool]],
) -> pd.DataFrame:
    """Interpolate between the unshaded and shaded input values, using the total irradiance and sky
        view as proportional values for each point.

    Args:
        unshaded (HourlyContinuousCollection):
            A collection of hourly values for the unshaded case.
        shaded (HourlyContinuousCollection):
            A collection of hourly values for the shaded case.
        total_irradiance (pd.DataFrame):
            A dataframe with the total irradiance for each point for each hour.
        sky_view (pd.DataFrame):
            A dataframe with the sky view for each point.
        sun_up_bool (np.ndarray):
            A list if booleans stating whether the sun is up.

    Returns:
        pd.DataFrame:
            A dataframe containing interpolated values, corresponding with the proportion of shade
            for each value in the input total_irradiance.
    """

    y_original = np.stack([shaded.values, unshaded.values], axis=1)
    new_min = y_original[sun_up_bool].min(axis=1)
    new_max = y_original[sun_up_bool].max(axis=1)

    # DAYTIME
    irradiance = total_irradiance[sun_up_bool]
    irradiance_range = np.vstack([irradiance.min(), irradiance.max()]).T
    old_min = irradiance_range.min(axis=1)
    old_max = irradiance_range.max(axis=1)
    old_value = irradiance.values
    with np.errstate(divide="ignore", invalid="ignore"):
        daytime = pd.DataFrame(
            (
                (((old_value - old_min) / (old_max - old_min)).T * (new_max - new_min))
                + new_min
            ).T
        )
    daytime.index = total_irradiance[sun_up_bool].index
    daytime.columns = total_irradiance[sun_up_bool].columns

    # NIGHTTIME
    x_original = [0, 100]
    nighttime = pd.DataFrame(
        interp1d(x_original, y_original[~sun_up_bool])(sky_view.values)[:, :, 0]
    )
    nighttime.index = total_irradiance[~sun_up_bool].index
    nighttime.columns = total_irradiance.columns

    interpolated_result = (
        pd.concat([nighttime, daytime], axis=0)
        .sort_index()
        .interpolate()
        .ewm(span=1.5)
        .mean()
    )

    return interpolated_result
