from ladybug.datacollection import HourlyContinuousCollection
import pandas as pd

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")

from LadybugTools.index_from_leap_year_bool import index_from_leap_year_bool
from LadybugTools.get_title_str import get_title_str


def hourlycontinuouscollection_to_series(collection: HourlyContinuousCollection) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.
    
    Args:
        collection (HourlyContinuousCollection): A ladybug data collection.

    Returns:
        str: A Pandas Series.
    """

    index = index_from_leap_year_bool(collection.header.analysis_period.is_leap_year)
    name = get_title_str(collection)
    return pd.Series(data=list(collection.values), index=index, name=name)
