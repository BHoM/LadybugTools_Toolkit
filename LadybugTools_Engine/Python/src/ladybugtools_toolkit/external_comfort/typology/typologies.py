from enum import Enum
from .typology import Typology
from ..shelter import Shelter


class Typologies(Enum):
    Openfield = Typology(
        name="Openfield",
        evaporative_cooling_effectiveness=0,
        shelters=[],
        wind_speed_multiplier=1,
    )
    Enclosed = Typology(
        name="Enclosed",
        evaporative_cooling_effectiveness=0,
        shelters=[Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0)],
        wind_speed_multiplier=1,
    )
    PorousEnclosure = Typology(
        name="Porous enclosure",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[0, 360], porosity=0.5)
        ],
        wind_speed_multiplier=1,
    )
    SkyShelter = Typology(
        name="Sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0)],
        wind_speed_multiplier=1,
    )
    FrittedSkyShelter = Typology(
        name="Fritted sky-shelter",
        evaporative_cooling_effectiveness=0,
        shelters=[
            Shelter(altitude_range=[45, 90], azimuth_range=[0, 360], porosity=0.5),
        ],
        wind_speed_multiplier=1,
    )
    NearWater = Typology(
        name="Near water",
        evaporative_cooling_effectiveness=0.15,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ],
        wind_speed_multiplier=1.2,
    )
    Misting = Typology(
        name="Misting",
        evaporative_cooling_effectiveness=0.3,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ],
        wind_speed_multiplier=0.5,
    )
    PDEC = Typology(
        name="PDEC",
        evaporative_cooling_effectiveness=0.7,
        shelters=[
            Shelter(altitude_range=[0, 0], azimuth_range=[0, 0], porosity=1),
        ],
        wind_speed_multiplier=0.5,
    )
    NorthShelter = Typology(
        name="North shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[337.5, 22.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    NortheastShelter = Typology(
        name="Northeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[22.5, 67.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    EastShelter = Typology(
        name="East shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    SoutheastShelter = Typology(
        name="Southeast shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[112.5, 157.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    SouthShelter = Typology(
        name="South shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[157.5, 202.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    SouthwestShelter = Typology(
        name="Southwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[202.5, 247.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    WestShelter = Typology(
        name="West shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    NorthwestShelter = Typology(
        name="Northwest shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[292.5, 337.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    NorthShelterWithCanopy = Typology(
        name="North shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[337.5, 22.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    NortheastShelterWithCanopy = Typology(
        name="Northeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[22.5, 67.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    EastShelterWithCanopy = Typology(
        name="East shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    SoutheastShelterWithCanopy = Typology(
        name="Southeast shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[112.5, 157.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    SouthShelterWithCanopy = Typology(
        name="South shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[157.5, 202.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    SouthwestShelterWithCanopy = Typology(
        name="Southwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[202.5, 247.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    WestShelterWithCanopy = Typology(
        name="West shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    NorthwestShelterWithCanopy = Typology(
        name="Northwest shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[292.5, 337.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    EastWestShelter = Typology(
        name="East-west shelter",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 70], azimuth_range=[67.5, 112.5], porosity=0),
            Shelter(altitude_range=[0, 70], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
    EastWestShelterWithCanopy = Typology(
        name="East-west shelter (with canopy)",
        evaporative_cooling_effectiveness=0.0,
        shelters=[
            Shelter(altitude_range=[0, 90], azimuth_range=[67.5, 112.5], porosity=0),
            Shelter(altitude_range=[0, 90], azimuth_range=[247.5, 292.5], porosity=0),
        ],
        wind_speed_multiplier=1,
    )
