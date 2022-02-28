import sys
from typing import List, Union

import numpy as np

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from ladybug.sunpath import Sunpath, Sun
from ladybug.epw import EPW


def sun_positions(epw: EPW) -> List[Sun]:
    """
    Calculate sun positions for a given epw file.

    Args:
        epw: An epw object.
    Returns:
        A list of Sun objects.
    """
    sunpath = Sunpath.from_location(epw.location)

    return [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]


def sun_altitudes_from_epw(epw: EPW) -> List[float]:
    """
    Calculate sun altitudes for a given epw file.

    Args:
        epw: An epw object.
    Returns:
        A list of sun altitudes.
    """
    return [sun.altitude for sun in sun_positions(epw)]


def sun_azimuths_from_epw(epw: EPW) -> List[float]:
    """
    Calculate sun azimuths for a given epw file.

    Args:
        epw: An epw object.
    Returns:
        A list of sun azimuths.
    """
    return [sun.azimuth for sun in sun_positions(epw)]


def sun_altitudes_from_suns(suns: List[Sun]) -> List[float]:
    """
    Calculate sun altitudes for a given list of sun objects.

    Args:
        suns: A list of sun objects.
    Returns:
        A list of sun altitudes.
    """
    return [sun.altitude for sun in suns]


def sun_azimuths_from_suns(suns: List[Sun]) -> List[float]:
    """
    Calculate sun azimuths for a given list of sun objects.

    Args:
        suns: A list of sun objects.
    Returns:
        A list of sun azimuths.
    """
    return [sun.azimuth for sun in suns]


def sun_altitudes(input: Union[List[Sun], EPW]) -> List[float]:
    if isinstance(input, EPW):
        return sun_altitudes_from_epw(input)
    elif isinstance(input, list) and isinstance(input[0], Sun):
        return sun_altitudes_from_suns(input)
    else:
        raise ValueError("Input must be an EPW object or a list of Sun objects.")


def sun_azimuths(input: Union[List[Sun], EPW]) -> List[float]:
    if isinstance(input, EPW):
        return sun_azimuths_from_epw(input)
    elif isinstance(input, list) and isinstance(input[0], Sun):
        return sun_azimuths_from_suns(input)
    else:
        raise ValueError("Input must be an EPW object or a list of Sun objects.")

def sun_positions_as_array(epw: EPW) -> np.ndarray([4, 2], dtype=float):
    """Return the sun location as an array of [altitude, azimuth]."""
    
    suns = sun_positions(epw)
    return np.array([[sun.altitude, sun.azimuth] for sun in suns])


if __name__ == "__main__":
    epw = EPW(
        r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw"
    )
    suns = sun_positions(epw)
    print(sun_altitudes(suns)[:24], sun_azimuths(epw)[:24])
