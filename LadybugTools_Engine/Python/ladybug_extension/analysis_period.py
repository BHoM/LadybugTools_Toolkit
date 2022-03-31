
from datetime import datetime
from typing import List

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
