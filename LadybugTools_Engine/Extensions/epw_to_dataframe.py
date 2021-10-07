import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW

import sys
sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.psychrometrics_dataframe import psychrometrics_dataframe
from LadybugTools.sun_position_dataframe import sun_position_dataframe
from LadybugTools.index_from_leap_year_bool import index_from_leap_year_bool


def epw_to_dataframe(epw: EPW) -> pd.DataFrame:
    """Convert an EPW object into a Pandas DataFrame
    Args:
        epw (EPW): A Ladybug EPW object.
    Returns:
        pd.DataFrame: A Pandas DataFrame containing the EPW contents in tabular form.
    """

    epw.dry_bulb_temperature  # make __slots__ accessible

    df = pd.DataFrame(index=index_from_leap_year_bool(epw.is_leap_year))
    for prop in dir(epw):
        try:
            cv = getattr(epw, prop)
            if isinstance(cv, HourlyContinuousCollection):
                df[prop] = cv.values
        except ValueError as e:
            df[prop] = [0] * 8760

    # Add sun positions
    df = pd.concat([df, sun_position_dataframe(epw)], axis=1)

    # Add psychrometrics
    df = pd.concat([df, psychrometrics_dataframe(epw)], axis=1)

    return df
