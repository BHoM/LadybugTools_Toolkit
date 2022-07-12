from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.time import Time
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.sunpath import Sunpath


def equation_of_time(epw: EPW) -> HourlyContinuousCollection:
    """Calculate the equation of time for each hour of the year.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of equation of times.
    """
    sunpath = Sunpath.from_location(epw.location)

    _, eot = list(
        zip(
            *[
                list(sunpath._calculate_solar_geometry(i))
                for i in epw.dry_bulb_temperature.datetimes
            ]
        )
    )

    return HourlyContinuousCollection(
        Header(
            data_type=Time(name="Equation of Time"),
            unit="min",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        eot,
    )
