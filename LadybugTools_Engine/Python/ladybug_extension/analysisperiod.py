import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod


def to_datetimes(self: AnalysisPeriod) -> pd.DatetimeIndex:
    """Convert an AnalysisPeriod object into a Pandas DatetimeIndex."""
    return pd.to_datetime(self.datetimes)

if __name__ == "__main__":
    pass