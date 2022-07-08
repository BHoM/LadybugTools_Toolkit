import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybug.wea import Wea


def radiation_tilt_orientation_matrix(epw: EPW) -> pd.DataFrame:
    """Compute the annual cumulative radiation matrix per surface tilt and orientation, for a given EPW object.

    Args:
        epw (EPW): The EPW object for which this calculation is made.

    Returns:
        pd.DataFrame: _description_
    """

    wea = Wea.from_annual_values(
        epw.location,
        epw.direct_normal_radiation.values,
        epw.diffuse_horizontal_radiation.values,
        is_leap_year=epw.is_leap_year,
    )

    # I implement a bit of a hack here, to calculate only the Eastern insolation - then mirror it about the North-South axis to get the whole matrix
    altitudes = np.linspace(0, 90, 10)
    azimuths = np.linspace(0, 180, 19)
    combinations = np.array(list(itertools.product(altitudes, azimuths)))

    def f(alt_az):
        return wea.__copy__().directional_irradiance(alt_az[0], alt_az[1])[0].total

    with ThreadPoolExecutor() as executor:
        results = np.array([i for i in executor.map(f, combinations[0:])]).reshape(
            len(altitudes), len(azimuths)
        )

    temp = pd.DataFrame(results, index=altitudes, columns=azimuths)

    new_cols = (360 - temp.columns)[::-1][1:]
    new_vals = temp.values[::-1, ::-1][
        ::-1, 1:
    ]  # some weird array transformation stuff here
    mirrored = pd.DataFrame(new_vals, columns=new_cols, index=temp.index)
    return pd.concat([temp, mirrored], axis=1)
