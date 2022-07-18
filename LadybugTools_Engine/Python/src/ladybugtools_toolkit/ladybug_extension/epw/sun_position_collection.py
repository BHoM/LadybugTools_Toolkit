from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.generic import GenericType
from ladybug.epw import EPW
from ladybug.header import Header

from ladybugtools_toolkit.ladybug_extension.epw.sun_position_list import (
    sun_position_list,
)


def sun_position_collection(epw: EPW) -> HourlyContinuousCollection:
    """Calculate a set of Sun positions for each hour of the year.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of sun positions.
    """

    suns = sun_position_list(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=GenericType(name="Sun Position", unit="Sun"),
            unit="Sun",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        suns,
    )
