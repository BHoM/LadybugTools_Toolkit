from ladybug.sunpath import Sunpath
from ladybug.location import Location
from ..bhom.analytics import bhom_analytics

import pandas as pd
from datetime import datetime

@bhom_analytics()
def sunpath_metadata(sunpath: Sunpath) -> dict:
    """Return a dictionary containing equinox and solstice altitudes at sunrise, noon and sunset for the given location

    Args:
        location (Location):
            A Ladybug location object.

    Returns:
        dict:
            A dictionary containing the altitudes in the following structure:
            
            {
                'december_solstice': {'sunrise': , 'noon': altitude, 'sunset': altitude},
                'march_equinox': {...},
                'june_solstice': {...},
                'september_equinox': {...}
            }
    """
    december_solstice_times = sunpath.calculate_sunrise_sunset_from_datetime(datetime(2023, 12, 22))
    march_equinox_times = sunpath.calculate_sunrise_sunset_from_datetime(datetime(2023, 3, 20))
    june_solstice_times = sunpath.calculate_sunrise_sunset_from_datetime(datetime(2023, 6, 21))
    september_equinox_times = sunpath.calculate_sunrise_sunset_from_datetime(datetime(2023, 9, 22))
    
    december_solstice = {k:{"time": v, "azimuth": sunpath.calculate_sun_from_date_time(v).azimuth} if k in ["sunrise", "sunset"] else {"time": v, "altitude": sunpath.calculate_sun_from_date_time(v).altitude} for k, v in december_solstice_times.items() }
    
    march_equinox = {k:{"time": v, "azimuth": sunpath.calculate_sun_from_date_time(v).azimuth} if k in ["sunrise", "sunset"] else {"time": v, "altitude": sunpath.calculate_sun_from_date_time(v).altitude} for k, v in march_equinox_times.items() }
    
    june_solstice = {k:{"time": v, "azimuth": sunpath.calculate_sun_from_date_time(v).azimuth} if k in ["sunrise", "sunset"] else {"time": v, "altitude": sunpath.calculate_sun_from_date_time(v).altitude} for k, v in june_solstice_times.items() }
    
    september_equinox = {k:{"time": v, "azimuth": sunpath.calculate_sun_from_date_time(v).azimuth} if k in ["sunrise", "sunset"] else {"time": v, "altitude": sunpath.calculate_sun_from_date_time(v).altitude} for k, v in september_equinox_times.items() }
    
    return {
        "december_solstice": december_solstice,
        "march_equinox": march_equinox,
        "june_solstice": june_solstice,
        "september_equinox": september_equinox
        }


