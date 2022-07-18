import numpy as np
import pandas as pd
from honeybee_energy.material._base import _EnergyMaterialOpaqueBase
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.model.create_model import create_model
from ladybugtools_toolkit.external_comfort.simulate.mean_radiant_temperature_collections import (
    mean_radiant_temperature_collections,
)
from ladybugtools_toolkit.external_comfort.typology import Typology
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.helpers.decay_rate_smoother import (
    decay_rate_smoother,
)


def effective_mean_radiant_temperature(
    typology: Typology,
    epw: EPW,
    ground_material: _EnergyMaterialOpaqueBase,
    shade_material: _EnergyMaterialOpaqueBase,
    identifier: str = None,
) -> HourlyContinuousCollection:
    """Return the effective mean radiant temperature for the given typology following a
        simulation.

    Args:
        epw (EPW): The EPW file in which to simulate the model.
        ground_material (_EnergyMaterialOpaqueBase): A surface material for the ground zones
            topmost face.
        shade_material (_EnergyMaterialOpaqueBase): A surface material for the shade zones
            faces.
        identifier (str, optional): A unique identifier for the model. Defaults to None which
            will generate a unique identifier. This is useful for testing purposes!

    Returns:
        HourlyContinuousCollection: An calculated mean radiant temperature based on the shelter
            configuration for the given typology.
    """

    mrt_collection = mean_radiant_temperature_collections(
        create_model(ground_material, shade_material, identifier), epw
    )

    shaded_mrt = to_series(mrt_collection["shaded_mean_radiant_temperature"])
    unshaded_mrt = to_series(mrt_collection["unshaded_mean_radiant_temperature"])

    daytime = np.array([i > 0 for i in epw.global_horizontal_radiation])
    sun_exposure = typology.sun_exposure(epw)
    mrts = []
    for hour in range(8760):
        if daytime[hour]:
            mrts.append(
                np.interp(
                    sun_exposure[hour],
                    [0, 1],
                    [shaded_mrt[hour], unshaded_mrt[hour]],
                )
            )
        else:
            mrts.append(
                np.interp(
                    typology.sky_exposure(),
                    [0, 1],
                    [shaded_mrt[hour], unshaded_mrt[hour]],
                )
            )

    # Fill any gaps where sun-visible/sun-occluded values are missing, and apply an
    # exponentially weighted moving average to account for transition betwen shaded/unshaded
    # periods.
    mrt_series = pd.Series(
        mrts, index=shaded_mrt.index, name=shaded_mrt.name
    ).interpolate()

    mrt_series = decay_rate_smoother(
        mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
    )

    return from_series(mrt_series)
