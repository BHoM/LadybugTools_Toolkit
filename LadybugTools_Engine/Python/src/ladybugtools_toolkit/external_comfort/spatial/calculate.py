from typing import List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ...ladybug_extension.analysis_period import AnalysisPeriod, to_datetimes


def shaded_unshaded_interpolation(
    unshaded_value: List[float],
    shaded_value: List[float],
    total_irradiance: List[List[float]],
    sky_view: List[float],
    sun_up: List[bool],
) -> pd.DataFrame:
    """Interpolate between the unshaded and shaded input values, using the total irradiance and sky
        view as proportional values for each point.

    Args:
        unshaded (List[float]):
            A list of hourly values for the unshaded case.
        shaded (List[float]):
            A collection of hourly values for the shaded case.
        total_irradiance (List[List[float]]):
            An array with the total irradiance for each point for each hour.
        sky_view (List[float]):
            A list with the sky view for each point.
        sun_up_bool (List[bool]):
            A list of booleans stating whether the sun is up.

    Returns:
        pd.DataFrame:
            A dataframe containing interpolated values, corresponding with the proportion of shade
            for each value in the input total_irradiance.
    """

    # check for annual-hourly-ness
    if len(sun_up) != 8760:
        raise NotImplementedError(
            "This functionality is not yet available for non annual/hourly datasets."
        )

    # convert to ndarrays (if needed)
    if not isinstance(unshaded_value, np.ndarray):
        unshaded_value = np.array(unshaded_value)
    if not isinstance(shaded_value, np.ndarray):
        shaded_value = np.array(shaded_value)
    if not isinstance(total_irradiance, np.ndarray):
        total_irradiance = np.array(total_irradiance)
    if not isinstance(sky_view, np.ndarray):
        sky_view = np.array(sky_view)
    if not isinstance(sun_up, np.ndarray):
        sun_up = np.array(sun_up)

    # Check for shape alignment
    if not len(sky_view) == total_irradiance.shape[1]:
        raise ValueError(
            f"Number of sky-view values must match length of total irradiance values ({len(sky_view)} != {total_irradiance.shape[1]})"
        )
    if (
        len(unshaded_value)
        != len(shaded_value)
        != total_irradiance.shape[0]
        != len(sun_up)
    ):
        raise ValueError(
            f"Number of points-in-time must match for unshaded, shaded, total_irradiance and sun_up values ({len(sky_view)} != {total_irradiance.shape[1]})"
        )

    # get the target range into which the value should fit following interpolation
    target_range = np.stack([shaded_value, unshaded_value], axis=1)
    target_low = target_range.min(axis=1)
    target_high = target_range.max(axis=1)

    # DAYTIME
    # get the original range for total irradiance for each timestep
    daytime_rad = total_irradiance[sun_up]
    daytime_low = daytime_rad.min(axis=1)
    daytime_high = daytime_rad.max(axis=1)

    # interpolate between known values to get target values
    with np.errstate(divide="ignore", invalid="ignore"):
        daytime_values = (
            ((daytime_rad.T - daytime_low) / (daytime_high - daytime_low))
            * (target_high[sun_up] - target_low[sun_up])
            + target_low[sun_up]
        ).T

    # NIGHTTIME
    with np.errstate(divide="ignore", invalid="ignore"):
        nighttime_values = interp1d([0, 100], target_range[~sun_up])(sky_view)

    # create datetime index for resultant dataframe
    index = to_datetimes(AnalysisPeriod())

    return (
        pd.concat(
            [
                pd.DataFrame(index=index[sun_up], data=daytime_values),
                pd.DataFrame(index=index[~sun_up], data=nighttime_values),
            ],
            axis=0,
        )
        .sort_index()
        .set_index(to_datetimes(AnalysisPeriod()))
        .interpolate()
        .ewm(span=1.5)
        .mean()
    )
