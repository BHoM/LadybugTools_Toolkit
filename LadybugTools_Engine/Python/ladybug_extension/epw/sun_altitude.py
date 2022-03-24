import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List
from ladybug.epw import EPW
from ladybug_extension.epw.sun_position import sun_position


def sun_altitude(epw: EPW) -> List[float]:
    """
    Calculate sun altitudes for a given epw file.

    Args:
        epw (EPW): An epw object.
    Returns:
        List[float]: A list of sun altitudes.
    """
    suns = sun_position(epw)
    return [sun.altitude for sun in suns]
