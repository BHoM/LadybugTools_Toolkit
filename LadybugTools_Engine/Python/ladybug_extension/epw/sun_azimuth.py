import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from ladybug.epw import EPW
from ladybug_extension.epw.sun_position import sun_position
from typing import List

def sun_azimuth(epw: EPW) -> List[float]:
    """
    Calculate sun azimuths for a given epw file.

    Args:
        epw (EPW): An epw object.
    Returns:
        List[float]: A list of sun azimuths.
    """
    suns = sun_position(epw)
    return [sun.azimuth for sun in suns]
