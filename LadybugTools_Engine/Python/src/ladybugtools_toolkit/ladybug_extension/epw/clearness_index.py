from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.fraction import Fraction
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.skymodel import clearness_index as lb_ci
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude import solar_altitude
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_collection import (
    sun_position_collection,
)


from python_toolkit.bhom.analytics import analytics


@analytics
def clearness_index(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate the clearness index value for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of clearness indices.
    """

    if not sun_position:
        sun_position = sun_position_collection(epw)

    ci = []
    for i, j, k in list(
        zip(
            *[
                epw.global_horizontal_radiation,
                solar_altitude(epw, sun_position),
                epw.extraterrestrial_direct_normal_radiation,
            ]
        )
    ):
        try:
            ci.append(lb_ci(i, j, k))
        except ZeroDivisionError:
            ci.append(0)

    return HourlyContinuousCollection(
        header=Header(
            data_type=Fraction(name="Clearness Index"),
            unit="fraction",
            analysis_period=AnalysisPeriod(),
        ),
        values=ci,
    )
