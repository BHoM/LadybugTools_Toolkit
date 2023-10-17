"""Module for creating typology objects."""
# pylint: disable=E0401
import inspect
import warnings

# pylint: enable=E0401

import numpy as np

from ._typologybase import Typology
from .shelter import (
    east_wall,
    east_west_linear,
    north_south_linear,
    north_wall,
    northeast_southwest_linear,
    northeast_wall,
    northwest_southeast_linear,
    northwest_wall,
    overhead_large,
    overhead_small,
    south_wall,
    southeast_wall,
    southwest_wall,
    west_wall,
)


def openfield() -> Typology:
    return Typology(
        name="Openfield",
    )


def enclosed() -> Typology:
    return Typology(
        name="Enclosed",
        shelters=[
            north_wall(),
            east_wall(),
            south_wall(),
            west_wall(),
            overhead_small(),
        ],
    )


def porous_enclosure() -> Typology:
    return Typology(
        name="Porous enclosure",
        shelters=[
            north_wall().set_porosity([0.5] * 8760),
            east_wall().set_porosity([0.5] * 8760),
            south_wall().set_porosity([0.5] * 8760),
            west_wall().set_porosity([0.5] * 8760),
            overhead_small().set_porosity([0.5] * 8760),
        ],
    )


def sky_shelter() -> Typology:
    return Typology(
        name="Sky-shelter",
        shelters=[
            overhead_large(),
        ],
    )


def fritted_sky_shelter() -> Typology:
    return Typology(
        name="Fritted sky-shelter",
        shelters=[
            overhead_large().set_porosity([0.5] * 8760),
        ],
    )


def near_water() -> Typology:
    return Typology(
        name="Near water",
        evaporative_cooling_effect=[0.15] * 8760,
    )


def misting() -> Typology:
    return Typology(
        name="Misting",
        evaporative_cooling_effect=[0.3] * 8760,
    )


def pdec() -> Typology:
    return Typology(
        name="PDEC",
        evaporative_cooling_effect=[0.7] * 8760,
    )


def north_shelter() -> Typology:
    return Typology(
        name="North shelter",
        shelters=[
            north_wall(),
        ],
    )


def northeast_shelter() -> Typology:
    return Typology(name="Northeast shelter", shelters=[northeast_wall()])


def east_shelter() -> Typology:
    return Typology(name="East shelter", shelters=[east_wall()])


def southeast_shelter() -> Typology:
    return Typology(name="Southeast shelter", shelters=[southeast_wall()])


def south_shelter() -> Typology:
    return Typology(
        name="South shelter",
        shelters=[
            south_wall(),
        ],
    )


def southwest_shelter() -> Typology:
    return Typology(name="Southwest shelter", shelters=[southwest_wall()])


def west_shelter() -> Typology:
    return Typology(name="West shelter", shelters=[west_wall()])


def northwest_shelter() -> Typology:
    return Typology(name="Northwest shelter", shelters=[northwest_wall()])


def north_shelter_with_canopy() -> Typology:
    return Typology(
        name="North shelter with canopy",
        shelters=[
            north_wall(),
            overhead_small(),
        ],
    )


def northeast_shelter_with_canopy() -> Typology:
    return Typology(
        name="Northeast shelter with canopy",
        shelters=[
            northeast_wall(),
            overhead_small(),
        ],
    )


def east_shelter_with_canopy() -> Typology:
    return Typology(
        name="East shelter with canopy",
        shelters=[
            east_wall(),
            overhead_small(),
        ],
    )


def southeast_shelter_with_canopy() -> Typology:
    return Typology(
        name="Southeast shelter with canopy",
        shelters=[
            southeast_wall(),
            overhead_small(),
        ],
    )


def south_shelter_with_canopy() -> Typology:
    return Typology(
        name="South shelter with canopy",
        shelters=[
            south_wall(),
            overhead_small(),
        ],
    )


def southwest_shelter_with_canopy() -> Typology:
    return Typology(
        name="Southwest shelter with canopy",
        shelters=[
            southwest_wall(),
            overhead_small(),
        ],
    )


def west_shelter_with_canopy() -> Typology:
    return Typology(
        name="West shelter with canopy",
        shelters=[
            west_wall(),
            overhead_small(),
        ],
    )


def northwest_shelter_with_canopy() -> Typology:
    return Typology(
        name="Northwest shelter with canopy",
        shelters=[
            northwest_wall(),
            overhead_small(),
        ],
    )


def north_south_linear_shelter() -> Typology:
    return Typology(
        name="North-south linear overhead shelter",
        shelters=[
            north_south_linear(),
        ],
    )


def northeast_southwest_linear_shelter() -> Typology:
    return Typology(
        name="Northeast-southwest linear overhead shelter",
        shelters=[
            northeast_southwest_linear(),
        ],
    )


def east_west_linear_shelter() -> Typology:
    return Typology(
        name="East-west linear overhead shelter",
        shelters=[
            east_west_linear(),
        ],
    )


def northwest_southeast_linear_shelter() -> Typology:
    return Typology(
        name="Northwest-southeast linear overhead shelter",
        shelters=[
            northwest_southeast_linear(),
        ],
    )


def combine_typologies(
    typologies: list[Typology],
    evaporative_cooling_effect_weights: list[float] = None,
    target_wind_speed_weights: list[float] = None,
    radiant_temperature_adjustment_weights: list[float] = None,
) -> Typology:
    """Combine multiple typologies into a single typology.

    Args:
        typologies (list[Typology]):
            A list of typologies to combine.
        evaporative_cooling_effect_weights (list[float], optional):
            A list of weights to apply to the evaporative cooling effect
            of each typology. Defaults to None.
        target_wind_speed_weights (list[float], optional):
            A list of weights to apply to the wind speed multiplier
            of each typology. Defaults to None.
        radiant_temperature_adjustment_weights (list[float], optional):
            A list of weights to apply to the radiant temperature adjustment
            of each typology. Defaults to None.

    Raises:
        ValueError: If the weights do not sum to 1.

    Returns:
        Typology: A combined typology.
    """

    all_shelters = []
    for typ in typologies:
        all_shelters.extend(typ.shelters)

    if np.isnan([i.target_wind_speed for i in typologies]).sum() != 0:
        warnings.warn(
            f"\n{inspect.stack()[0][3]}\n"
            "Some typologies do not have a target wind speed - at timesteps "
            "aligned with these, the wind speed from the EPW adjusted by "
            "shelter exposure will be used."
        )
    target_wind_speed_data = np.stack(
        [i.target_wind_speed for i in typologies],
        axis=0,
    )
    masked_data = np.ma.masked_array(
        target_wind_speed_data, np.isnan(target_wind_speed_data)
    )
    average_ws_data = np.ma.average(
        masked_data, axis=0, weights=target_wind_speed_weights
    )

    return Typology(
        name=" + ".join([i.name for i in typologies]),
        shelters=all_shelters,
        evaporative_cooling_effect=np.average(
            [i.evaporative_cooling_effect for i in typologies],
            weights=evaporative_cooling_effect_weights,
            axis=0,
        ).tolist(),
        target_wind_speed=average_ws_data.filled(np.nan).tolist(),
        radiant_temperature_adjustment=np.average(
            [i.radiant_temperature_adjustment for i in typologies],
            weights=radiant_temperature_adjustment_weights,
            axis=0,
        ).tolist(),
    )
