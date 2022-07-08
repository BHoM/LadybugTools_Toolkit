import numpy as np
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.angle import Angle
from ladybug.epw import EPW
from ladybug.header import Header

from .sun_position_collection import sun_position_collection


def apparent_solar_zenith(
    epw: EPW, solar_altitude: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly apparent solar zenith angles.

    Args:
        epw (EPW): An EPW object.
        solar_altitude (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of solar altitude angles. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of apparent solar zenith angles.
    """

    if not solar_altitude:
        solar_altitude = sun_position_collection(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=Angle(name="Apparent Solar Zenith"),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [np.pi / 2 - i for i in solar_altitude.values],
    )
