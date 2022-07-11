from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection

from .apparent_solar_zenith import apparent_solar_zenith


def apparent_solar_zenith_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly apparent solar zenith angle.

    Args:
        epw (EPW): An EPW object.
        sun_position (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of apparent solar zenith angles.
    """

    collection = apparent_solar_zenith(epw, sun_position)
    collection = collection.to_unit("radians")

    return collection
