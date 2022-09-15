from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.epw import EPW
from ladybug.psychrometrics import wet_bulb_from_db_rh


from python_toolkit.bhom.analytics import analytics


@analytics
def wet_bulb_temperature(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly wet bulb temperature for a given EPW.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of wet bulb temperatures.
    """
    return HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        WetBulbTemperature(),
        "C",
    )
