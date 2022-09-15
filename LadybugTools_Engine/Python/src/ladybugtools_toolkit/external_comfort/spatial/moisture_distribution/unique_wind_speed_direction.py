from typing import List, Tuple

import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


from python_toolkit.bhom.analytics import analytics


@analytics
def unique_wind_speed_direction(
    epw: EPW, schedule: List[int] = None
) -> List[Tuple[float, float]]:
    """Return a set of unique wind speeds and directions for an EPW file.

    Args:
        epw (EPW): An epw object.
        schedule (epw): a mask of hours to include in the unique

    Returns:
        List[List[float, float]]: A list of unique (wind_speed, wind_direction).
    """

    df = pd.concat([to_series(epw.wind_speed), to_series(epw.wind_direction)], axis=1)

    if schedule is not None:
        df = df.iloc[schedule]

    return df.drop_duplicates().values
