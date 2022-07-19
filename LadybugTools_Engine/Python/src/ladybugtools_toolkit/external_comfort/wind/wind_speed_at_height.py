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
        wind_shear_exponent = 1 / 7
        return reference_wind_speed * (
            np.pow((target_height / reference_height), wind_shear_exponent)
        )
