import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.typology import Typology
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def effective_wind_speed(typology: Typology, epw: EPW) -> HourlyContinuousCollection:
    """Calculate wind speed when subjected to a set of shelters.

    Args:
        typology (Typology): A Typology object.
        epw (EPW): The input EPW.

    Returns:
        HourlyContinuousCollection: The resultant wind-speed.
    """

    if len(typology.shelters) == 0:
        return epw.wind_speed

    collections = []
    for shelter in typology.shelters:
        collections.append(to_series(shelter.effective_wind_speed(epw)))
    return from_series(
        pd.concat(collections, axis=1).min(axis=1).rename("Wind Speed (m/s)")
    )
