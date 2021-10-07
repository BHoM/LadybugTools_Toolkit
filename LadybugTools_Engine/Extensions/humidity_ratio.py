from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.psychrometrics import humid_ratio_from_db_rh
from ladybug.datatype.fraction import HumidityRatio


def humidity_ratio(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly humidity ratio collection for a given EPW.

    Args:
        epw (EPW): A ladybug EPW object.

    Returns:
        HourlyContinuousCollection: A Humidity Ratio data collection.
    """
    humidity_ratio = HourlyContinuousCollection.compute_function_aligned(
        humid_ratio_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        HumidityRatio(),
        "fraction",
    )
    return humidity_ratio
