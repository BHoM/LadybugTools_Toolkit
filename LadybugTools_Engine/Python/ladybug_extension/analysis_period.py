from datetime import datetime
from typing import List
import calendar

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod

from ladybug_extension.dt import from_datetime


def to_datetimes(analysis_period: AnalysisPeriod) -> pd.DatetimeIndex:
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
        from_datetime(min(datetimes)), from_datetime(max(datetimes)), inferred_timestep
    )

    if len(analysis_period.datetimes) != len(datetimes):
        raise ValueError(
            f"The number of datetimes ({len(datetimes)}) does not match the number of datetimes in the AnalysisPeriod ({len(analysis_period.datetimes)})."
        )

    return analysis_period


def describe_analysis_period(
    analysis_period: AnalysisPeriod, save_path: bool = False
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod): A Ladybug analysis period.
        save_path (str, optional): If True, create a path-safe string from the analysis period.

    Returns:
        str: A description of the analysis period.
    """

    if save_path:
        return f"{analysis_period.st_month:02}{analysis_period.st_day:02}_{analysis_period.end_month:02}{analysis_period.end_day:02}_{analysis_period.st_hour:02}_{analysis_period.end_hour:02}"
    else:
        return f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to {calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between {analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:00"
