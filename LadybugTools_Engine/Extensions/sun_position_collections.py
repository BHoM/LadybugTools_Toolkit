from typing import Dict

import numpy as np
from ladybug.datacollection import AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.datatype.angle import Angle
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath


def sun_position_collections(epw: EPW) -> Dict[str, HourlyContinuousCollection]:
    """Calculate annual hourly sun positions for a given EPW.

    Args:
        epw (EPW): A ladybug EPW object.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing solar_azimuth, solar_altitude and apparent_solar_zenith in radians.
    """
    sunpath = Sunpath.from_location(epw.location)
    sp = [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]
    solar_azimuth = HourlyContinuousCollection(
        Header(
            data_type=Angle(),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.azimuth_in_radians for i in sp],
    )
    solar_altitude = HourlyContinuousCollection(
        Header(
            data_type=Angle(),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.altitude_in_radians for i in sp],
    )
    apparent_solar_zenith = HourlyContinuousCollection(
        Header(
            data_type=Angle(),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [np.pi / 2 - i for i in solar_altitude.values],
    )
    return {
        "solar_azimuth": solar_azimuth,
        "solar_altitude": solar_altitude,
        "apparent_solar_zenith": apparent_solar_zenith,
    }
