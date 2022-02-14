from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection
import numpy as np


def create_windspeed_collection(
    epw: EPW, target_average_wind_speed: float
) -> HourlyContinuousCollection:
    """Create an annual hourly collection of windspeeds whose average equals the target value, translated to 10m height, using the source EPW to provide a wind-speed profile.

    Args:
        epw (EPW): The source EPW from which teh wind speed profile is used to distribute wind speeds.
        target_average_wind_speed (float): The value to be translated to 10m and set as the average for the target wind-speed collection.

    Returns:
        HourlyContinuousCollection: A ladybug annual hourly data wind speed collection.
    """

    # Translate target wind speed at ground level to wind speed at 10m, assuming an open terrain per airport conditions
    target_average_wind_speed_at_10m = wind_speed_at_height(
        target_average_wind_speed, 1.2, 10, 0.03
    )

    # Adjust hourly values in wind_speed to give a new overall average equal to that of the target wind-speed
    adjustment_factor = target_average_wind_speed_at_10m / epw.wind_speed.average

    return epw.wind_speed * adjustment_factor


def wind_speed_at_height(
    referenceWindSpeed: float,
    referenceHeight: float,
    targetHeight: float,
    terrainRoughnessLength: float,
    logFunction: bool = True,
) -> float:
    """Calculate the wind speed at a given height from the 10m default height as stated in an EPW file.

    Args:
        referenceWindSpeed (float): The speed to be translated.
        referenceHeight (float): The original height of the wind speed being translated.
        targetHeight (float): The target height of the wind speed being translated.
        terrainRoughnessLength (float): A value describing how rough the ground is.
        logFunction (bool, optional): Set to True to used the log transformation method, or False for the exponent method. Defaults to True.

    Returns:
        float: The translated wind speed at the target height.
    """
    if logFunction:
        return referenceWindSpeed * (
            np.log(targetHeight / terrainRoughnessLength)
            / np.log(referenceHeight / terrainRoughnessLength)
        )
    else:
        windShearExponent = 1 / 7
        return referenceWindSpeed * (
            np.pow((targetHeight / referenceHeight), windShearExponent)
        )
