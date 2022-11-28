import calendar
from datetime import datetime
from typing import List

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod

from .dt import from_datetime


def to_datetimes(
    analysis_period: AnalysisPeriod,
) -> pd.DatetimeIndex:
    """Convert an AnalysisPeriod object into a Pandas DatetimeIndex.

    Args:
        analysis_period (AnalysisPeriod): An AnalysisPeriod object.

    Returns:
        pd.DatetimeIndex: A Pandas DatetimeIndex object.
    """

    datetimes = pd.to_datetime(analysis_period.datetimes)

    return datetimes


def from_datetimes(datetimes: List[datetime]) -> AnalysisPeriod:
    """Convert a list of datetimes (in order from earliest to latest) into an AnalysisPeriod object.

    Args:
        datetimes (List[datetime]): A list of datetimes.

    Returns:
        AnalysisPeriod: An AnalysisPeriod object.
    """

    inferred_timestep = (60 * 60) / (datetimes[1] - datetimes[0]).seconds

    analysis_period = AnalysisPeriod.from_start_end_datetime(
        from_datetime(min(datetimes)),
        from_datetime(max(datetimes)),
        inferred_timestep,
    )

    if len(analysis_period.datetimes) != len(datetimes):
        raise ValueError(
            f"The number of datetimes ({len(datetimes)}) does not match the number of datetimes in "
            "the AnalysisPeriod ({len(analysis_period.datetimes)}), which probably means your "
            "datetime-list has an irregular time-step and cannot be coerced into an AnalysisPeriod."
        )
    return analysis_period


def describe(
    analysis_period: AnalysisPeriod,
    save_path: bool = False,
    include_timestep: bool = False,
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod): A Ladybug analysis period.
        save_path (bool, optional): If True, create a path-safe string from the analysis period.
        include_timestep (bool, optional): If True, include the timestep in the description.

    Returns:
        str: A description of the analysis period.
    """

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
        if include_timestep:
            return f"{analysis_period.st_month:02}{analysis_period.st_day:02}_{analysis_period.end_month:02}{analysis_period.end_day:02}_{analysis_period.st_hour:02}_{analysis_period.end_hour:02}_{analysis_period.timestep:02}"
        return f"{analysis_period.st_month:02}{analysis_period.st_day:02}_{analysis_period.end_month:02}{analysis_period.end_day:02}_{analysis_period.st_hour:02}_{analysis_period.end_hour:02}"

    if include_timestep:
        return f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to {calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between {analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:00, every {timestep[analysis_period.timestep]}"

    return f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to {calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between {analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:00"
