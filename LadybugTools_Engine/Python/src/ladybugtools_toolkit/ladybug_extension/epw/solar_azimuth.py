from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.angle import Angle
from ladybug.epw import EPW
from ladybug.header import Header
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_collection import (
    sun_position_collection,
)


from python_toolkit.bhom.analytics import analytics


@analytics
def solar_azimuth(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar azimuth angles.
    """

    if not sun_position:
        sun_position = sun_position_collection(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=Angle(
                name="Solar Azimuth",
            ),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.azimuth for i in sun_position.values],
    )
