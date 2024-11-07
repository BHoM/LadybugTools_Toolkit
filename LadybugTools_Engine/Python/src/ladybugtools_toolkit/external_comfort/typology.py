"""Module for creating typology objects."""

# pylint: disable=E0401,E1101
from enum import Enum

import numpy as np
from python_toolkit.bhom.analytics import bhom_analytics

from ._shelterbase import Shelter
from ._typologybase import Typology

# pylint: enable=E0401




class Typologies(Enum):
    """_"""

    OPENFIELD = Typology(identifier="Openfield")
    ENCLOSED = Typology(
        identifier="Fully enclosed",
        shelters=[
            Shelter.from_adjacent_wall(angle=0),
            Shelter.from_adjacent_wall(angle=90),
            Shelter.from_adjacent_wall(angle=180),
            Shelter.from_adjacent_wall(angle=270),
            Shelter.from_overhead_circle(),
        ],
    )
    ENCLOSED_POROUS = Typology(
        identifier="Fully enclosed 50% porous",
        shelters=[
            Shelter.from_adjacent_wall(angle=0).set_porosity(0.5),
            Shelter.from_adjacent_wall(angle=90).set_porosity(0.5),
            Shelter.from_adjacent_wall(angle=180).set_porosity(0.5),
            Shelter.from_adjacent_wall(angle=270).set_porosity(0.5),
            Shelter.from_overhead_circle().set_porosity(0.5),
        ],
    )
    SKY_SHELTER = Typology(
        identifier="Sky shelter",
        shelters=[
            Shelter.from_overhead_circle(radius=5),
        ],
    )
    NEAR_WATER = Typology(
        identifier="Near water",
        evaporative_cooling_effect=(0.15,) * 8760,
    )
    MISTING = Typology(
        identifier="Misting",
        evaporative_cooling_effect=(0.3,) * 8760,
    )
    PASSIVE_DOWNDRAUGHT_EVAPORATIVE_COOLING_TOWER = Typology(
        identifier="Passive downdraught evaporative cooling tower",
        evaporative_cooling_effect=(
            0.7,
        ) * 8760,
        shelters=[
            Shelter.from_overhead_circle(
                height_above_ground=5,
                radius=4)],
    )
    NORTH_SHELTER = Typology(
        identifier="Shelter from north",
        shelters=[Shelter.from_adjacent_wall(angle=0)],
    )
    EAST_SHELTER = Typology(
        identifier="Shelter from east",
        shelters=[Shelter.from_adjacent_wall(angle=90)],
    )
    SOUTH_SHELTER = Typology(
        identifier="Shelter from south",
        shelters=[Shelter.from_adjacent_wall(angle=180)],
    )
    WEST_SHELTER = Typology(
        identifier="Shelter from west",
        shelters=[Shelter.from_adjacent_wall(angle=270)],
    )
    LINEAR_EAST_WEST_SHELTER = Typology(
        identifier="Linear overhead shelter from east to west",
        shelters=[Shelter.from_overhead_linear(angle=90)],
    )
    LINEAR_NORTH_SOUTH_SHELTER = Typology(
        identifier="Linear overhead shelter from north to south",
        shelters=[Shelter.from_overhead_linear(angle=0)],
    )


@bhom_analytics()
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

    target_wind_speed_avg = np.stack(
        [i.target_wind_speed for i in typologies],
        axis=0,
    ).T
    target_wind_speed_avg[target_wind_speed_avg is None] = np.nan
    target_wind_speed_avg = target_wind_speed_avg.astype(float)
    masked_arr = np.ma.masked_invalid(target_wind_speed_avg)
    target_wind_speed_avg = (
        np.ma.average(masked_arr, axis=1, weights=target_wind_speed_weights)
        .filled(np.nan)
        .tolist()
    )
    target_wind_speed_avg = [None if np.isnan(
        i) else i for i in target_wind_speed_avg]

    radiant_temperature_adjustment_avg = np.average(
        [i.radiant_temperature_adjustment for i in typologies],
        weights=radiant_temperature_adjustment_weights,
        axis=0,
    ).tolist()

    evaporative_cooling_effect_avg = np.average(
        [i.evaporative_cooling_effect for i in typologies],
        weights=evaporative_cooling_effect_weights,
        axis=0,
    ).tolist()

    return Typology(
        identifier=" + ".join([i.identifier for i in typologies]),
        shelters=all_shelters,
        evaporative_cooling_effect=evaporative_cooling_effect_avg,
        target_wind_speed=target_wind_speed_avg,
        radiant_temperature_adjustment=radiant_temperature_adjustment_avg,
    )
