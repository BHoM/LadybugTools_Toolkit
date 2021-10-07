import pandas as pd
from ladybug.epw import EPW

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.sun_position_collections import sun_position_collections
from LadybugTools.hourlycontinuouscollection_to_series import hourlycontinuouscollection_to_series
from LadybugTools.index_from_leap_year_bool import index_from_leap_year_bool


def sun_position_dataframe(epw: EPW) -> pd.DataFrame:
    """Calculate annual hourly sun positions for a given EPW.

    Args:
        epw (EPW): A ladybug EPW object.

    Returns:
        pd.DataFrame: A pandas DataFrame containing solar_azimuth, solar_altitude and apparent_solar_zenith in radians.
    """

    d = sun_position_collections(epw)
    df = pd.DataFrame(index=index_from_leap_year_bool(epw.is_leap_year))
    for k, v in d.items():
        df[k] = hourlycontinuouscollection_to_series(v).values

    return df
