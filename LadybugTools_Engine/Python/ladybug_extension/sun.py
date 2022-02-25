import sys
from typing import List
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from ladybug.sunpath import Sunpath, Sun
from ladybug.epw import EPW

def suns(epw: EPW) -> List[Sun]:
    """
    Calculate sun positions for a given epw file.

    Args:
        epw: An epw object.
    Returns:
        A list of Sun objects.
    """
    sunpath = Sunpath.from_location(epw.location)
    
    return [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]

def sun_altitudes(epw: EPW) -> List[float]:
    """
    Calculate sun altitudes for a given epw file.

    Args:
        epw: An epw object.
    Returns:
        A list of sun altitudes.
    """
    return [sun.altitude for sun in suns(epw)]

def sun_azimuths(epw: EPW) -> List[float]:
    """
    Calculate sun azimuths for a given epw file.

    Args:
        epw: An epw object.
    Returns:
        A list of sun azimuths.
    """
    return [sun.azimuth for sun in suns(epw)]

if __name__ == "__main__":
    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    print(sun_altitudes(epw)[:24])
