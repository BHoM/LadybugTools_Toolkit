"""Methods for manipulating Ladybug analysis periods."""

# pylint: disable=E0401
import calendar
from datetime import datetime

# pylint: enable=E0401

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from python_toolkit.bhom.analytics import bhom_analytics
from .dt import lb_datetime_from_datetime


def analysis_period_to_datetimes(
    analysis_period: AnalysisPeriod,
) -> pd.DatetimeIndex:
    """Convert an AnalysisPeriod object into a Pandas DatetimeIndex.

    Args:
        analysis_period (AnalysisPeriod):
            An AnalysisPeriod object.

    Returns:
        pd.DatetimeIndex:
            A Pandas DatetimeIndex object.
    """

    datetimes = pd.to_datetime(analysis_period.datetimes)

    return datetimes


@bhom_analytics()
def analysis_period_to_boolean(
    analysis_periods: list[AnalysisPeriod] | AnalysisPeriod,
) -> list[bool]:
    """Convert an AnalysisPeriod object into a list of booleans where values
    within the Period are also within a default whole analysis period of the
    same interval.

    Args:
        analysis_periods (list[AnalysisPeriod]):
            A list of AnalysisPeriod objects.

    Returns:
        list[bool]:
            A list of booleans
    """

    if isinstance(analysis_periods, AnalysisPeriod):
        analysis_periods = [analysis_periods]

    # check timestep of each analysis period is the same
    if len(set(ap.timestep for ap in analysis_periods)) > 1:
        raise ValueError("All analysis periods must have the same timestep.")

    # remove duplicates from list
    analysis_periods = list(set(analysis_periods))

    # create a generic set of datetimes for the same timestep
    generic_datetimes = analysis_period_to_datetimes(
        AnalysisPeriod(timestep=analysis_periods[0].timestep)
    )

    # for each analysis period in analysis_periods, create a list of booleans
    # where values within the Period are also within a default whole analysis
    # period of the same interval
    bools = []
    for ap in analysis_periods:
        bools.append(generic_datetimes.isin(analysis_period_to_datetimes(ap)))

    return np.any(bools, axis=0)


@bhom_analytics()
def analysis_period_from_datetimes(datetimes: list[datetime]) -> AnalysisPeriod:
    """Convert a list of datetimes (in order from earliest to latest) into an
    AnalysisPeriod object.

    Args:
        datetimes (list[datetime]):
            qA list of datetimes.

    Returns:
        AnalysisPeriod:
            An AnalysisPeriod object.
    """

    inferred_timestep = (60 * 60) / (datetimes[1] - datetimes[0]).seconds

    analysis_period = AnalysisPeriod.from_start_end_datetime(
        lb_datetime_from_datetime(min(datetimes)),
        lb_datetime_from_datetime(max(datetimes)),
        inferred_timestep,
    )

    if len(analysis_period.datetimes) != len(datetimes):
        raise ValueError(
            f"The number of datetimes ({len(datetimes)}) does not match the number of datetimes in "
            "the AnalysisPeriod ({len(analysis_period.datetimes)}), which probably means your "
            "datetime-list has an irregular time-step and cannot be coerced into an AnalysisPeriod."
        )
    return analysis_period


@bhom_analytics()
def describe_analysis_period(
    analysis_period: list[AnalysisPeriod],
    save_path: bool = False,
    include_timestep: bool = False,
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.
        save_path (bool, optional):
            If True, create a path-safe string from the analysis period.
        include_timestep (bool, optional):
            If True, include the timestep in the description.

    Returns:
        str:
            A description of the analysis period.
    """

    if isinstance(analysis_period, AnalysisPeriod):
        analysis_period = [analysis_period]

    # remove duplicates from list
    analysis_period = list(set(analysis_period))

    # check timestep of each analysis period is the same
    if len(set(ap.timestep for ap in analysis_period)) > 1:
        raise ValueError("All analysis periods must have the same timestep.")

    timestep = {
        1: "hour",
        2: "half-hour",
        3: "20 minutes",
        4: "15 minutes",
        5: "12 minutes",
        6: "10 minutes",
        10: "6 minutes",
        12: "5 minutes",
        15: "4 minutes",
        20: "3 minutes",
        30: "2 minutes",
        60: "minute",
    }

    if save_path:
        if len(analysis_period) != 1:
            raise ValueError("Only one analysis period can be used for a save path.")
        analysis_period = analysis_period[0]
        if include_timestep:
            return (
                f"{analysis_period.st_month:02}{analysis_period.st_day:02}_"
                f"{analysis_period.end_month:02}{analysis_period.end_day:02}_"
                f"{analysis_period.st_hour:02}_{analysis_period.end_hour:02}_"
                f"{analysis_period.timestep:02}"
            )
        return (
            f"{analysis_period.st_month:02}{analysis_period.st_day:02}_"
            f"{analysis_period.end_month:02}{analysis_period.end_day:02}_"
            f"{analysis_period.st_hour:02}_{analysis_period.end_hour:02}"
        )

    base_str = []
    for ap in analysis_period:
        base_str.append(
            f"{calendar.month_abbr[ap.st_month]} {ap.st_day:02} to "
            f"{calendar.month_abbr[ap.end_month]} {ap.end_day:02} between "
            f"{ap.st_hour:02}:00 and {ap.end_hour:02}:59"
        )
    base_str = ", and ".join(base_str)

    if include_timestep:
        return f"{base_str}, every {timestep[analysis_period[0].timestep]}"

    return base_str


def analysis_period_from_description(description_str: str) -> AnalysisPeriod:
    # TODO: implement this, based on the output from the method above to recreate the analysis period
    raise NotImplementedError()


@bhom_analytics()
def do_analysis_periods_represent_entire_year(
    analysis_periods: list[AnalysisPeriod],
) -> bool:
    """Check a list of analysis periods to see if they represent an entire year.

    Args:
        analysis_periods (list[AnalysisPeriod]):
            A list of analysis periods.

    Returns:
        bool:
            True if the analysis periods represent an entire year, errors are
            raised otherwise.

    """
    if any(ap.end_hour < ap.st_hour for ap in analysis_periods):
        raise ValueError(
            "To combine time periods crossing midnight, AnalysisPeriod should "
            "be split into two parts - one for either side of midnight."
        )

    # Validation
    if any(ap.timestep != 1 for ap in analysis_periods):
        raise ValueError("All input analysis period timesteps must be hourly.")

    if any(
        ap.is_leap_year != analysis_periods[0].is_leap_year for ap in analysis_periods
    ):
        raise ValueError(
            "All input analysis periods must be either leap year, or not leap "
            "year. Mixed leapedness is not allowed."
        )

    target_datetimes = analysis_period_to_datetimes(AnalysisPeriod())
    actual_datetimes = (
        pd.concat(
            [analysis_period_to_datetimes(ap).to_series() for ap in analysis_periods]
        )
        .sort_index()
        .index
    )
    target_timesteps = 8784 if analysis_periods[0].is_leap_year else 8760
    actual_timesteps = sum(len(ap) for ap in analysis_periods)
    if actual_timesteps > target_timesteps:
        duplicates = actual_datetimes[actual_datetimes.duplicated()]
        raise ValueError(
            "The number of timesteps contained within the input analysis "
            f"periods is greater than {target_timesteps}. Duplicate timesteps "
            f"are {duplicates}"
        )
    if actual_timesteps < target_timesteps:
        # pylint: disable=E1125
        missing = (
            pd.DatetimeIndex(list(set(target_datetimes) - set(actual_datetimes)))
            .to_series()
            .sort_index()
            .index
        )
        # pylint: enable=E1125
        raise ValueError(
            "The number of timesteps contained within the input analysis "
            f"periods is less than {target_timesteps}. Missing timesteps "
            f"are {missing}"
        )

    return True
