from datetime import datetime

from ladybug.sunpath import Sunpath
from ladybugtools_toolkit.ladybug_extension.sunpath import \
    sunrise_sunset_azimuths


def sunpath_metadata(sunpath: Sunpath) -> dict:
    """Return a dictionary containing equinox and solstice azimuths and altitudes at sunrise, noon and sunset for the given sunpath.

    Args:
        sunpath (Sunpath):
            A Ladybug sunpath object.

    Returns:
        dict:
            A dictionary containing the azimuths and altitudes in the following structure:

            {
                'december_solstice': {'sunrise': azimuth, 'noon': altitude, 'sunset': azimuth},
                'march_equinox': {...},
                'june_solstice': {...},
                'september_equinox': {...}
            }
    """

    december_solstice = sunrise_sunset_azimuths(sunpath, 2023, 12, 22)
    march_equinox = sunrise_sunset_azimuths(sunpath, 2023, 3, 20)
    june_solstice = sunrise_sunset_azimuths(sunpath, 2023, 6, 21)
    september_equinox = sunrise_sunset_azimuths(sunpath, 2023, 9, 22)

    return {
        "december_solstice": december_solstice,
        "march_equinox": march_equinox,
        "june_solstice": june_solstice,
        "september_equinox": september_equinox,
    }
