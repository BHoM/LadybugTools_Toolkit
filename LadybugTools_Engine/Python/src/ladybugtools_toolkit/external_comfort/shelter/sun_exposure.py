from typing import List

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.shelter import Shelter
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_list import (
    sun_position_list,
)


def sun_exposure(shelters: List[Shelter], epw: EPW) -> List[float]:
    """Return NaN if sun below horizon, and a value between 0-1 for sun-hidden to sun-exposed.

    Args:
        shelters (List[Shelter]): Shelters that could block the sun.
        epw (EPW): An EPW object.

    Returns:
        List[float]: A list of sun visibility values for each hour of the year.
    """

    suns = sun_position_list(epw)
    sun_is_up = np.array([sun.altitude > 0 for sun in suns])

    nans = np.empty(len(epw.dry_bulb_temperature))
    nans[:] = np.NaN

    if len(shelters) == 0:
        return np.where(sun_is_up, 1, nans)

    blocked = []
    for shelter in shelters:
        temp = np.where(shelter.sun_blocked(suns), shelter.porosity, nans)
        temp = np.where(np.logical_and(np.isnan(temp), sun_is_up), 1, temp)
        blocked.append(temp)

    return pd.DataFrame(blocked).T.min(axis=1).values.tolist()
