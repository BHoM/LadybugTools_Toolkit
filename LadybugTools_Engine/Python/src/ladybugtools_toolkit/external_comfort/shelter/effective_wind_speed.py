from typing import List

import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW

from ...ladybug_extension.datacollection import from_series, to_series
from .shelter import Shelter


def effective_wind_speed(
    shelters: List[Shelter], epw: EPW
) -> HourlyContinuousCollection:
    """Calculate wind speed when subjected to a set of shelters.

    Args:
        shelters (List[Shelter]): A list of Shelter objects.
        epw (EPW): The input EPW.

    Returns:
        HourlyContinuousCollection: The resultant wind-speed.
    """

    if len(shelters) == 0:
        return epw.wind_speed

    collections = []
    for shelter in shelters:
        collections.append(to_series(shelter.effective_wind_speed(epw)))
    return from_series(
        pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
    )
