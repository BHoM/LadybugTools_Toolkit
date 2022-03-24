import sys

from external_comfort.wind.wind_speed_at_height import wind_speed_at_height

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW


def target_wind_speed_collection(
    epw: EPW, target_average_wind_speed: float, target_height: float
) -> HourlyContinuousCollection:
    """Create an annual hourly collection of windspeeds whose average equals the target value, translated to 10m height, using the source EPW to provide a wind-speed profile.

    Args:
        epw (EPW): The source EPW from which the wind speed profile is used to distribute wind speeds.
        target_average_wind_speed (float): The value to be translated to 10m and set as the average for the target wind-speed collection.
        target_height (float): The height at which the wind speed is translated to (this will assume the original wind speed is at 10m per EPW conventions.

    Returns:
        HourlyContinuousCollection: A ladybug annual hourly data wind speed collection.
    """

    # Translate target wind speed at ground level to wind speed at 10m, assuming an open terrain per airport conditions
    target_average_wind_speed_at_10m = wind_speed_at_height(
        target_average_wind_speed, target_height, 10, 0.03
    )

    # Adjust hourly values in wind_speed to give a new overall average equal to that of the target wind-speed
    adjustment_factor = target_average_wind_speed_at_10m / epw.wind_speed.average

    return epw.wind_speed * adjustment_factor

