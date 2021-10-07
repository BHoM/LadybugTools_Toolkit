from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.psychrometrics import wet_bulb_from_db_rh


def wet_bulb_temperature(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly wet bulb temperature collection for a given EPW.

    Args:
        epw (EPW): A ladybug EPW object.

    Returns:
        HourlyContinuousCollection: A Wet Bulb Temperature data collection.
    """
    wet_bulb_temperature = HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        WetBulbTemperature(),
        "C",
    )
    return wet_bulb_temperature
