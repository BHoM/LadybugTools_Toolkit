"""Methods for determining the Koeppen climate classification of a location."""

# pylint: disable=E0401
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# pylint: enable=E0401

import pandas as pd
from ladybug.epw import EPW, Location
from scipy import spatial


@dataclass
class KoeppenClassification:
    """Summary of KoeppenClassification dataclass."""

    classification: str = field(init=True, default=None)
    description: str = field(init=True, default=None)
    group: str = field(init=True, default=None)
    precipitation_type: str = field(init=True, default=None)
    level_of_heat: str = field(init=True, default=None)

    def __post_init__(self):
        classifications = [
            "Af",
            "Am",
            "As",
            "Aw",
            "BSh",
            "BSk",
            "BWh",
            "BWk",
            "Cfa",
            "Cfb",
            "Cfc",
            "Csa",
            "Csb",
            "Csc",
            "Cwa",
            "Cwb",
            "Cwc",
            "Dfa",
            "Dfb",
            "Dfc",
            "Dfd",
            "Dsa",
            "Dsb",
            "Dsc",
            "Dwa",
            "Dwb",
            "Dwc",
            "Dwd",
            "EF",
            "ET",
        ]
        if self.classification not in classifications:
            raise ValueError(f"Invalid Koeppen classification: {self.classification}")

    def to_dict(self):
        """Return a dictionary representation of the KoeppenClassification."""
        return {
            "classification": self.classification,
            "description": self.description,
            "group": self.group,
            "precipitation_type": self.precipitation_type,
            "level_of_heat": self.level_of_heat,
        }


class KoeppenClimateClassifications(Enum):
    """Summary of KoeppenClimateClassifications enum."""

    AF = KoeppenClassification(
        classification="Af",
        description="Tropical rainforest climate",
        group="Tropical",
        precipitation_type="Rainforest",
        level_of_heat=None,
    )
    AM = KoeppenClassification(
        classification="Am",
        description="Tropical monsoon climate",
        group="Tropical",
        precipitation_type="Monsoon",
        level_of_heat=None,
    )
    AS = KoeppenClassification(
        classification="As",
        description="Tropical dry savanna climate",
        group="Tropical",
        precipitation_type="Savanna, Dry",
        level_of_heat=None,
    )
    AW = KoeppenClassification(
        classification="Aw",
        description="Tropical savanna, wet",
        group="Tropical",
        precipitation_type="Savanna, Wet",
        level_of_heat=None,
    )
    BSH = KoeppenClassification(
        classification="BSh",
        description="Hot semi-arid (steppe) climate",
        group="Arid",
        precipitation_type="Steppe",
        level_of_heat="Hot",
    )
    BSK = KoeppenClassification(
        classification="BSk",
        description="Cold semi-arid (steppe) climate",
        group="Arid",
        precipitation_type="Steppe",
        level_of_heat="Cold",
    )
    BWH = KoeppenClassification(
        classification="BWh",
        description="Hot deserts climate",
        group="Arid",
        precipitation_type="Desert",
        level_of_heat="Hot",
    )
    BWK = KoeppenClassification(
        classification="BWk",
        description="Cold desert climate",
        group="Arid",
        precipitation_type="Desert",
        level_of_heat="Cold",
    )
    CFA = KoeppenClassification(
        classification="Cfa",
        description="Humid subtropical climate",
        group="Temperate",
        precipitation_type="Without dry season",
        level_of_heat="Hot summer",
    )
    CFB = KoeppenClassification(
        classification="Cfb",
        description="Temperate oceanic climate",
        group="Temperate",
        precipitation_type="Without dry season",
        level_of_heat="Warm summer",
    )
    CFC = KoeppenClassification(
        classification="Cfc",
        description="Subpolar oceanic climate",
        group="Temperate",
        precipitation_type="Without dry season",
        level_of_heat="Cold summer",
    )
    CSA = KoeppenClassification(
        classification="Csa",
        description="Hot-summer Mediterranean climate",
        group="Temperate",
        precipitation_type="Dry summer",
        level_of_heat="Hot summer",
    )
    CSB = KoeppenClassification(
        classification="Csb",
        description="Warm-summer Mediterranean climate",
        group="Temperate",
        precipitation_type="Dry summer",
        level_of_heat="Warm summer",
    )
    CSC = KoeppenClassification(
        classification="Csc",
        description="Cool-summer Mediterranean climate",
        group="Temperate",
        precipitation_type="Dry summer",
        level_of_heat="Cold summer",
    )
    CWA = KoeppenClassification(
        classification="Cwa",
        description="Monsoon-influenced humid subtropical climate",
        group="Temperate",
        precipitation_type="Dry winter",
        level_of_heat="Hot summer",
    )
    CWB = KoeppenClassification(
        classification="Cwb",
        description="Subtropical highland climate or temperate oceanic climate with dry winters",
        group="Temperate",
        precipitation_type="Dry winter",
        level_of_heat="Warm summer",
    )
    CWC = KoeppenClassification(
        classification="Cwc",
        description="Cold subtropical highland climate or subpolar oceanic climate with dry winters",
        group="Temperate",
        precipitation_type="Dry winter",
        level_of_heat="Cold summer",
    )
    DFA = KoeppenClassification(
        classification="Dfa",
        description="Hot-summer humid continental climate",
        group="Cold (continental)",
        precipitation_type="Without dry season",
        level_of_heat="Hot summer",
    )
    DFB = KoeppenClassification(
        classification="Dfb",
        description="Warm-summer humid continental climate",
        group="Cold (continental)",
        precipitation_type="Without dry season",
        level_of_heat="Warm summer",
    )
    DFC = KoeppenClassification(
        classification="Dfc",
        description="Subarctic climate",
        group="Cold (continental)",
        precipitation_type="Without dry season",
        level_of_heat="Cold summer",
    )
    DFD = KoeppenClassification(
        classification="Dfd",
        description="Extremely cold subarctic climate",
        group="Cold (continental)",
        precipitation_type="Without dry season",
        level_of_heat="Very cold winter",
    )
    DSA = KoeppenClassification(
        classification="Dsa",
        description="Hot, dry-summer continental climate",
        group="Cold (continental)",
        precipitation_type="Dry summer",
        level_of_heat="Hot summer",
    )
    DSB = KoeppenClassification(
        classification="Dsb",
        description="Warm, dry-summer continental climate",
        group="Cold (continental)",
        precipitation_type="Dry summer",
        level_of_heat="Warm summer",
    )
    DSC = KoeppenClassification(
        classification="Dsc",
        description="Dry-summer subarctic climate",
        group="Cold (continental)",
        precipitation_type="Dry summer",
        level_of_heat="Cold summer",
    )
    DWA = KoeppenClassification(
        classification="Dwa",
        description="Monsoon-influenced hot-summer humid continental climate",
        group="Cold (continental)",
        precipitation_type="Dry winter",
        level_of_heat="Hot summer",
    )
    DWB = KoeppenClassification(
        classification="Dwb",
        description="Monsoon-influenced warm-summer humid continental climate",
        group="Cold (continental)",
        precipitation_type="Dry winter",
        level_of_heat="Warm summer",
    )
    DWC = KoeppenClassification(
        classification="Dwc",
        description="Monsoon-influenced subarctic climate",
        group="Cold (continental)",
        precipitation_type="Dry winter",
        level_of_heat="Cold summer",
    )
    DWD = KoeppenClassification(
        classification="Dwd",
        description="Monsoon-influenced extremely cold subarctic climate",
        group="Cold (continental)",
        precipitation_type="Dry winter",
        level_of_heat="Very cold winter",
    )
    EF = KoeppenClassification(
        classification="EF",
        description="Ice cap climate",
        group="Polar",
        precipitation_type="Ice cap",
        level_of_heat=None,
    )
    ET = KoeppenClassification(
        classification="ET",
        description="Tundra",
        group="Polar",
        precipitation_type="Tundra",
        level_of_heat=None,
    )


def koeppen_classification(
    arg: EPW | Location | Path | tuple[float],
) -> KoeppenClassification:
    """For the given EPW object, return a dict containing Koeppen climate classification data

    Args:koeppenAS
        arg (EPW | Location | Path | tuple[float]):
            An EPW object, or a tuple of (latitude, longitude) values, or a string or Path to an EPW file.

    Returns:
        KoeppenClassification: An object containing details for the resultant Koeppen climate type.
    """

    if isinstance(arg, (str, Path)):
        epw = EPW(arg)
        location = epw.location
    elif isinstance(arg, EPW):
        location = arg.location
    elif isinstance(arg, Location):
        location = arg
    elif isinstance(arg, tuple):
        location = Location(latitude=arg[0], longitude=arg[1])
    else:
        raise TypeError(
            "Expected an EPW object, a tuple of (latitude, longitude) values, "
            f"or a string or Path to an EPW file. Got {type(arg)}"
        )

    # Load koeppen climate lookup dataset
    koeppen_location_lookup = (
        Path(__file__).parents[2] / "data" / "koeppen.csv"
    ).absolute()
    koeppen_locations = pd.read_csv(koeppen_location_lookup)
    locs = koeppen_locations[["Lat", "Lon"]].values

    # Find nearest koeppen climate classification
    pt = [location.latitude, location.longitude]
    _, index = spatial.KDTree(locs).query(pt)
    koeppen_class = koeppen_locations.Cls[index]

    return getattr(KoeppenClimateClassifications, koeppen_class.upper())
