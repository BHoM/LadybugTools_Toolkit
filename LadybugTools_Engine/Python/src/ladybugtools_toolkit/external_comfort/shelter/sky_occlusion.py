import numpy as np


def sky_occlusion(
    start_altitude: float, end_altitude: float, width: float, porosity: float = 0
) -> float:
    """Determine the proportion of the sky occluded by a given spherical patch.

    Args:
        start_altitude (float): The altitude of the lowest part of the patch (in radians).
        end_altitude (float): The altitude of the highest part of the patch (in radians).
        width (float): The width of the patch (in radians).
        porosity (float): An amoutn applied to the occluded patch to account for visibility
            *through* it due to porosity. Default is 0, meaning fully opaque.

    Returns: The proportion of sky occluded by the spherical patch
    """

    if width < 0:
        raise ValueError("You cannot occlude the sky with a negatively sized patch.")

    if any(
        [
            start_altitude < 0,
            start_altitude > np.pi / 2,
            end_altitude < 0,
            end_altitude > np.pi / 2,
        ]
    ):
        raise ValueError("Start and end altitudes must be between 0 and pi/2.")

    area_occluded = abs(
        (np.sin(end_altitude) - np.sin(start_altitude)) * width / (2 * np.pi)
    )

    return area_occluded * (1 - porosity)
