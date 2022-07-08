from ladybug.sunpath import Sunpath
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.time import Time
from ladybug.datacollection import HourlyContinuousCollection

from .equation_of_time import equation_of_time as eot


def solar_time_hour(
    epw: EPW, equation_of_time: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (in hour-of-day) for each hour of the year.

    Args:
        epw (EPW): An EPW object.
        equation_of_time (HourlyContinuousCollection, optional): A pre-calculated equation of time HourlyContinuousCollection. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar time (in hour-of-day).
    """

    if equation_of_time is None:
        equation_of_time = eot(epw)

    sunpath = Sunpath.from_location(epw.location)
    hour_values = [i.hour for i in epw.dry_bulb_temperature.datetimes]

    solar_time = [
        sunpath._calculate_solar_time(j, k, False)
        for j, k in list(zip(*[hour_values, equation_of_time.values]))
    ]

    return HourlyContinuousCollection(
        Header(
            data_type=Time(name="Solar Time"),
            unit="hr",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        solar_time,
    )
