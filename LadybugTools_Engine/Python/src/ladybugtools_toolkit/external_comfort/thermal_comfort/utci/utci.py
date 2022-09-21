from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug_comfort.collection.utci import UTCI
from ladybugtools_toolkit.external_comfort.thermal_comfort.utci.utci_vectorised import (
    utci_vectorised,
)
from numpy.typing import NDArray


from ladybugtools_toolkit import analytics


@analytics
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

    if all((isinstance(i, HourlyContinuousCollection) for i in _inputs)):
        return UTCI(
            air_temperature=air_temperature,
            rel_humidity=relative_humidity,
            rad_temperature=mean_radiant_temperature,
            wind_speed=wind_speed,
        ).universal_thermal_climate_index

    if all((isinstance(i, (float, int)) for i in _inputs)):
        return utci_vectorised(
            ta=air_temperature,
            rh=relative_humidity,
            tr=mean_radiant_temperature,
            vel=np.clip([wind_speed], 0, 17)[0],
        )

    if all((isinstance(i, pd.DataFrame) for i in _inputs)):
        return pd.DataFrame(
            utci_vectorised(
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

    if all((isinstance(i, pd.Series) for i in _inputs)):
        return pd.Series(
            utci_vectorised(
                ta=air_temperature,
                rh=relative_humidity,
                tr=mean_radiant_temperature,
                vel=wind_speed.clip(lower=0, upper=17),
            ),
            name="Universal Thermal Climate Index (C)",
            index=_inputs[0].index,
        )

    if all((isinstance(i, (List, Tuple)) for i in _inputs)):
        return utci_vectorised(
            ta=np.array(air_temperature),
            rh=np.array(relative_humidity),
            tr=np.array(mean_radiant_temperature),
            vel=np.clip(np.array(wind_speed), 0, 17),
        )

    raise ValueError(
        "No possible means of calculating UTCI from that combination of inputs was found."
    )
