from ladybug.sunpath import Sunpath
from ladybug.location import Location
from ..bhom.analytics import bhom_analytics

import pandas as pd
from datetime import datetime

@bhom_analytics()
def sunrise_sunset_azimuths(sunpath: Sunpath, year: int, month: int, day: int) -> dict:
    """Return a dictionary containing azimuths at sunrise and sunset, and altitude at solar noon
    
    Args:
        sunpath (Sunpath):
            a ladybug sunpath object
        
        month (int):
            month as an int, starting at 1 and ending at 12
        
        day (int):
            day as an int, starting at 1, and ending at 28 through 31 depending on the month
    
    Returns:
        dict:
            A dictionary containing the azimuths and altitude in the following structure:
            {
                "sunrise": {"time": sunrise time, "azimuth": azimuth at sunrise },
                "noon": {"time": noon time, "altitude" altitude at noon },
                "sunset": {"time": sunset time, "azimuth": azimuth at sunset },
            }
    
    """
    
    sunrise_sunset = sunpath.calculate_sunrise_sunset_from_datetime(datetime(year=year, month=month, day=day))
    
    azimuths_altitude =  {k:{"time": datetime(year, v.month, v.day, v.hour, v.minute), "azimuth": sunpath.calculate_sun_from_date_time(v).azimuth} if k in ["sunrise", "sunset"] else {"time": datetime(year, v.month, v.day, v.hour, v.minute), "altitude": sunpath.calculate_sun_from_date_time(v).altitude} for k, v in sunrise_sunset.items() }
    
    return azimuths_altitude