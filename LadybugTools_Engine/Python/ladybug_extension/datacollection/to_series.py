import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import pandas as pd
from ladybug._datacollectionbase import BaseCollection
from ladybug_extension.analysisperiod.to_datetimes import to_datetimes
from ladybug_extension.header.to_string import to_string


def to_series(collection: BaseCollection) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.
    
    Args:
        collection: Ladybug data collection object.
        
    Returns:
        pd.Series: A Pandas Series object.
    """

    index = to_datetimes(collection.header.analysis_period)
    if len(collection.values) == 12:
        index = pd.date_range(f"{index[0].year}-01-01", periods=12, freq="MS")

    return pd.Series(
        data=collection.values,
        index=index,
        name=to_string(collection.header),
    )
