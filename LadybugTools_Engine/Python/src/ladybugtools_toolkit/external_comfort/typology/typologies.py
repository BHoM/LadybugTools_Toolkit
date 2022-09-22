from enum import Enum

from ladybugtools_toolkit.external_comfort.shelter.shelter import Shelter
from ladybugtools_toolkit.external_comfort.typology.typology import Typology


class Typologies(Enum):
    """A list of pre-defined Typology objects."""

    OPENFIELD = Typology(
        name="Openfield",
        evaporative_cooling_effectiveness=0,
        shelters=[],
    )
    ENCLOSED = Typology(
        name="Enclosed",
        evaporative_cooling_effectiveness=0,
        shelters=[Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], wind_porosity=0, radiation_porosity=0)],
        wind_speed_adjustment=1,
    )
    POROUS_ENCLOSURE = Typology(
        name="Porous enclosure",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], wind_porosity=0, radiation_porosity=0.5)
        ],
        wind_speed_adjustment=1,
    )
    SKY_SHELTER = Typology(
        name="Sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], wind_porosity=0, radiation_porosity=0)],
        wind_speed_adjustment=1,
    )
    FRITTED_SKY_SHELTER = Typology(
        name="Fritted sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], wind_porosity=0.5, radiation_porosity=0.5),
        ],
        wind_speed_adjustment=1,
    )
    NEAR_WATER = Typology(
        name="Near water",
        evaporative_cooling_effectiveness=0.15,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], radiation_porosity=1, wind_porosity=1),
        ],
        wind_speed_adjustment=1,
    )
    MISTING = Typology(
        name="Misting",
        evaporative_cooling_effectiveness=0.3,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], radiation_porosity=1, wind_porosity=1),
        ],
        wind_speed_adjustment=1,
    )
    PDEC = Typology(
        name="PDEC",
        evaporative_cooling_effectiveness=0.7,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], radiation_porosity=1, wind_porosity=1),
        ],
        wind_speed_adjustment=1,
    )
    NORTH_SHELTER = Typology(
        name="North shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[337.5, 22.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    NORTHEAST_SHELTER = Typology(
        name="Northeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[22.5, 67.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    EAST_SHELTER = Typology(
        name="East shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[67.5, 112.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHEAST_SHELTER = Typology(
        name="Southeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[112.5, 157.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    SOUTH_SHELTER = Typology(
        name="South shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[157.5, 202.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHWEST_SHELTER = Typology(
        name="Southwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[202.5, 247.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    WEST_SHELTER = Typology(
        name="West shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[247.5, 292.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    NORTHWEST_SHELTER = Typology(
        name="Northwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[292.5, 337.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    NORTH_SHELTER_WITH_CANOPY = Typology(
        name="North shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[337.5, 22.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    NORTHEAST_SHELTER_WITH_CANOPY = Typology(
        name="Northeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[22.5, 67.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    EAST_SHELTER_WITH_CANOPY = Typology(
        name="East shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[67.5, 112.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHEAST_SHELTER_WITH_CANOPY = Typology(
        name="Southeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[112.5, 157.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    SOUTH_SHELTER_WITH_CANOPY = Typology(
        name="South shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[157.5, 202.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    SOUTHWEST_SHELTER_WITH_CANOPY = Typology(
        name="Southwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[202.5, 247.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    WEST_SHELTER_WITH_CANOPY = Typology(
        name="West shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[247.5, 292.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    NORTHWEST_SHELTER_WITH_CANOPY = Typology(
        name="Northwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[292.5, 337.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    EAST_WEST_SHELTER = Typology(
        name="East-west shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[67.5, 112.5], radiation_porosity=0, wind_porosity=0),
            Shelter(altitude_range=[0, 70], azimuth_range=[247.5, 292.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
    EAST_WEST_SHELTER_WITH_CANOPY = Typology(
        name="East-west shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[67.5, 112.5], radiation_porosity=0, wind_porosity=0),
            Shelter(altitude_range=[0, 90], azimuth_range=[247.5, 292.5], radiation_porosity=0, wind_porosity=0),
        ],
        wind_speed_adjustment=1,
    )
