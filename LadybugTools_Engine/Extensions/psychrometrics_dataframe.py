import pandas as pd
from ladybug.epw import EPW

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.psychrometrics_collections import psychrometrics_collections
from LadybugTools.hourlycontinuouscollection_to_series import hourlycontinuouscollection_to_series
from LadybugTools.index_from_leap_year_bool import index_from_leap_year_bool


def psychrometrics_dataframe(epw: EPW) -> pd.DataFrame:
    """Generate a dictionary of calculated psychrometric collections.

    Args:
        epw (EPW): A ladybug EPW object.

    Returns:
        pd.DataFrame: A pandas DataFrame containing psychrometric data collections.
    """

    d = psychrometrics_collections(epw)
    df = pd.DataFrame(index=index_from_leap_year_bool(epw.is_leap_year))
    for k, v in d.items():
        df[k] = hourlycontinuouscollection_to_series(v).values

    return df
