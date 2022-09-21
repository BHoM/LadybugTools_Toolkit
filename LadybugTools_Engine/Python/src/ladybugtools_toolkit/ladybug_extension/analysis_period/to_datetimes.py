import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybugtools_toolkit import analytics


@analytics
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
