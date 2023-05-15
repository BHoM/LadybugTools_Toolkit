from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ...ladybug_extension.analysis_period import (
    AnalysisPeriod,
    analysis_period_to_datetimes,
)


def shaded_unshaded_interpolation(
    unshaded_value: Tuple[float],
    shaded_value: Tuple[float],
    total_irradiance: Tuple[Tuple[float]],
    sky_view: Tuple[float],
    sun_up: Tuple[bool],
) -> pd.DataFrame:
    """Interpolate between the unshaded and shaded input values, using the total irradiance and sky
        view as proportional values for each point.

    Args:
        unshaded_value (Tuple[float]):
            A list of hourly values for the unshaded case.
        shaded_value (Tuple[float]):
            A collection of hourly values for the shaded case.
        total_irradiance (Tuple[Tuple[float]]):
            An array with the total irradiance for each point for each hour.
        sky_view (Tuple[float]):
            A list with the sky view for each point.
        sun_up (Tuple[bool]):
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
    index = analysis_period_to_datetimes(AnalysisPeriod())

    return (
        pd.concat(
            [
                pd.DataFrame(index=index[sun_up], data=daytime_values),
                pd.DataFrame(index=index[~sun_up], data=nighttime_values),
            ],
            axis=0,
        )
        .sort_index()
        .set_index(analysis_period_to_datetimes(AnalysisPeriod()))
        .interpolate()
        .ewm(span=1.5)
        .mean()
    )


def rwdi_london_thermal_comfort_category(
    utci: pd.DataFrame,
    comfort_limits: Tuple[float] = (0, 32),
    hours: Tuple[float] = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
) -> pd.Series:
    """From a dataframe containing annual hourly spatial UTCI values,
    categorise each column as one of the RWDI London Thermal Comfort
    Guidelines categories.

    Method used here from City of London (2020), Thermal Comfort Guidelines
    for developments in the City of London, RWDI. URL:
    https://www.cityoflondon.gov.uk/services/planning/microclimate-guidelines
    [accessed on 2022-12-03].

    Args:
        utci (pd.DataFrame):
            A temporo-spatial UTCI data collection.
        comfort_limits (Tuple[float], optional):
            The UTCI values within which "comfortable" is defined.
            Defaults to (0, 32).
        hours (Tuple[float], optional):
            The hours to include in this assessment.
            Defaults to (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20).

    Returns:
        pd.Series:
            A series of comfort categories (one per column in the original
            dataframe).
    """

    if not isinstance(utci, pd.DataFrame):
        raise ValueError(
            "This method will not work with anything other than a pandas DataFrame."
        )

    if len(utci.index) != 8760:
        raise ValueError("Timesteps in the input dataframe != 8760.")

    # filter by time
    temp = utci[utci.index.hour.isin(hours)]

    # get comfort bool matrix
    comfort = (temp >= min(comfort_limits)) & (temp <= max(comfort_limits))

    # determine whether "Summer" is months [6, 7, 8] or [12, 1, 2],
    # and create season masks
    monthly_mean = utci.max(axis=1).groupby(utci.index.month).max()
    if monthly_mean[[12, 1, 2]].mean() < monthly_mean[[6, 7, 8]].mean():
        spring_mask = comfort.index.month.isin([3, 4, 5])
        summer_mask = comfort.index.month.isin([6, 7, 8])
        fall_mask = comfort.index.month.isin([9, 10, 11])
        winter_mask = comfort.index.month.isin([12, 1, 2])
    else:
        spring_mask = comfort.index.month.isin([9, 10, 11])
        summer_mask = comfort.index.month.isin([12, 1, 2])
        fall_mask = comfort.index.month.isin([3, 4, 5])
        winter_mask = comfort.index.month.isin([6, 7, 8])

    # calculate seasonal comfort percentatges
    spring_comfort = comfort[spring_mask].sum() / spring_mask.sum()
    summer_comfort = comfort[summer_mask].sum() / summer_mask.sum()
    fall_comfort = comfort[fall_mask].sum() / fall_mask.sum()
    winter_comfort = comfort[winter_mask].sum() / winter_mask.sum()

    # construct mask for difference categories
    transient = (winter_comfort < 0.25) | (
        (spring_comfort < 0.5) | (summer_comfort < 0.5) | (fall_comfort < 0.5)
    )
    short_term_seasonal = (winter_comfort >= 0.25) & (
        (spring_comfort >= 0.5) & (summer_comfort >= 0.5) & (fall_comfort >= 0.5)
    )
    short_term = (
        (winter_comfort >= 0.5)
        & (spring_comfort >= 0.5)
        & (summer_comfort >= 0.5)
        & (fall_comfort >= 0.5)
    )
    seasonal = (winter_comfort >= 0.7) & (
        (spring_comfort >= 0.9) & (summer_comfort >= 0.9) & (fall_comfort >= 0.9)
    )
    all_season = (
        (winter_comfort >= 0.9)
        & (spring_comfort >= 0.9)
        & (summer_comfort >= 0.9)
        & (fall_comfort >= 0.9)
    )

    # build series of categories
    comfort_category = pd.Series(
        np.where(
            all_season,
            "All Season",
            np.where(
                seasonal,
                "Seasonal",
                np.where(
                    short_term,
                    "Short-term",
                    np.where(
                        short_term_seasonal,
                        "Short-term Seasonal",
                        np.where(transient, "Transient", "Undefined"),
                    ),
                ),
            ),
        )
    )

    return comfort_category
