import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import numpy as np
from typing import List


def mean_radiant_temperature_from_surfaces(
    surface_temperatures: List[float], view_factors: List[float]
) -> float:
    """Calculate Mean Radiant Temperature from a list of surface temperature and view factors to those surfaces.

    Args:
        surface_temperatures (List[float]): A list of surface temperatures.
        view_factors (List[float]): A list of view-factors (one per surface)

    Returns:
        float: A value describing resultant radiant temperature.
    """

    if len(surface_temperatures) != len(view_factors):
        raise ValueError("The number of surface temperatures and view factors must be the same.")
    
    resultant_temperature = 0
    for i, temp in enumerate(surface_temperatures):
        temperature_kelvin = temp + 273.15
        resultant_temperature = (
            resultant_temperature + np.pow(temperature_kelvin, 4) * view_factors[i]
        )
    mean_radiant_temperature_kelvin = np.pow(resultant_temperature, 0.25)
    mean_radiant_temperature = mean_radiant_temperature_kelvin - 273.15
    return mean_radiant_temperature
