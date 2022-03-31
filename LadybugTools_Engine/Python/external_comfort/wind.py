

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
import numpy as np

def wind_speed_at_height(
    reference_wind_speed: float,
    reference_height: float,
    target_height: float,
    terrain_roughness_length: float,
    log_function: bool = True,
) -> float:
    """Calculate the wind speed at a given height from the 10m default height as stated in an EPW file.

    Args:
        reference_wind_speed (float): The speed to be translated.
        reference_height (float): The original height of the wind speed being translated.
        target_height (float): The target height of the wind speed being translated.
        terrain_roughness_length (float): A value describing how rough the ground is.
        log_function (bool, optional): Set to True to used the log transformation method, or False for the exponent method. Defaults to True.

    Returns:
        float: The translated wind speed at the target height.
    """
    if log_function:
        return reference_wind_speed * (
            np.log(target_height / terrain_roughness_length)
            / np.log(reference_height / terrain_roughness_length)
        )
    else:
        windShearExponent = 1 / 7
        return reference_wind_speed * (
            np.pow((target_height / reference_height), windShearExponent)
        )


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

