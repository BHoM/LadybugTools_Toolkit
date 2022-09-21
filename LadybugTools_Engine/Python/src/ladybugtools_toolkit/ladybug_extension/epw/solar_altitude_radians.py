from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude import solar_altitude


from ladybugtools_toolkit import analytics


@analytics
def solar_altitude_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar altitude angles.
    """

    collection = solar_altitude(epw, sun_position)
    collection = collection.to_unit("radians")

    return collection
