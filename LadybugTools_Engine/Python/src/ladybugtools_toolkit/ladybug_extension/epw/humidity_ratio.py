from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.fraction import HumidityRatio
from ladybug.epw import EPW
from ladybug.psychrometrics import humid_ratio_from_db_rh


def humidity_ratio(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly humidity ratio for a given EPW.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of humidity ratios.
    """
    return HourlyContinuousCollection.compute_function_aligned(
        humid_ratio_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        HumidityRatio(),
        "fraction",
    )
