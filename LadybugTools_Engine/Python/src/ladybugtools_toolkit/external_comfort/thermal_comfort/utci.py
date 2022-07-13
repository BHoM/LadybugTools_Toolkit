from typing import Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug_comfort.collection.utci import UTCI
from numpy.typing import NDArray

from .utci_vectorised import v_utci


def utci(
    air_temperature: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
    relative_humidity: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
    mean_radiant_temperature: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
    wind_speed: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
) -> Union[HourlyContinuousCollection, pd.DataFrame, NDArray[np.float64]]:
    """Return the UTCI for the given inputs.

    Returns:
        HourlyContinuousCollection: The calculated UTCI based on the shelter configuration for the given typology.
    """
    _inputs = [
        air_temperature,
        relative_humidity,
        mean_radiant_temperature,
        wind_speed,
    ]

    if not all(isinstance(item, type(_inputs[0])) for item in _inputs[1:]):
        try:
            [float(i) for i in _inputs]
        except TypeError:
            raise TypeError("All inputs must be numeric and of similar shape!")

    if isinstance(_inputs[0], HourlyContinuousCollection):
        return UTCI(
            air_temperature=air_temperature,
            rel_humidity=relative_humidity,
            rad_temperature=mean_radiant_temperature,
            wind_speed=wind_speed,
        ).universal_thermal_climate_index
    elif isinstance(_inputs[0], pd.DataFrame):
        return pd.DataFrame(
            v_utci(
                ta=air_temperature.values,
                rh=relative_humidity.values,
                tr=mean_radiant_temperature.values,
                vel=wind_speed.clip(lower=0, upper=17).values,
            ),
            columns=_inputs[0].columns
            if len(_inputs[0].columns) > 1
            else ["Universal Thermal Climate Index (C)"],
            index=_inputs[0].index,
        )
    elif isinstance(_inputs[0], pd.Series):
        return pd.Series(
            v_utci(
                ta=air_temperature,
                rh=relative_humidity,
                tr=mean_radiant_temperature,
                vel=wind_speed.clip(lower=0, upper=17),
            ),
            name="Universal Thermal Climate Index (C)",
            index=_inputs[0].index,
        )
    else:
        return v_utci(
            ta=air_temperature,
            rh=relative_humidity,
            tr=mean_radiant_temperature,
            vel=np.clip(wind_speed, 0, 17),
        )
