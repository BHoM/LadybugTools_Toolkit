import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod


def to_datetimes(self: AnalysisPeriod) -> pd.DatetimeIndex:
    """Convert an AnalysisPeriod object into a Pandas DatetimeIndex."""
    return pd.to_datetime(self.datetimes)
