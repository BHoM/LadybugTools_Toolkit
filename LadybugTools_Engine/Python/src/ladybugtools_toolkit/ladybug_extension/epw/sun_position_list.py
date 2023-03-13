from typing import List

from ladybug.epw import EPW
from ladybug.sunpath import Sun, Sunpath


from ladybugtools_toolkit import analytics


@analytics
def sun_position_list(epw: EPW) -> List[Sun]:
    """
    Calculate sun positions for a given epw file.

    Args:
        epw (EPW):
            An epw object.
    Returns:
        List[Sun]:
            A list of Sun objects.
    """

    sunpath = Sunpath.from_location(epw.location)

    return [sunpath.calculate_sun_from_hoy(i) for i in range(len(epw.years))]