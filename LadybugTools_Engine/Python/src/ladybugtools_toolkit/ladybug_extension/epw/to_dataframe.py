import pandas as pd
from ladybug.epw import EPW

from ..datacollection import to_series
from ..datacollection.monthlycollection import to_hourly
from .apparent_solar_zenith import apparent_solar_zenith as asz
from .clearness_index import clearness_index as ci
from .enthalpy import enthalpy as ent
from .equation_of_time import equation_of_time as eot
from .humidity_ratio import humidity_ratio as hr
from .solar_altitude import solar_altitude as sa
from .solar_altitude_radians import solar_altitude_radians as sar
from .solar_azimuth import solar_azimuth as saz
from .solar_azimuth_radians import solar_azimuth_radians as sazr
from .solar_declination import solar_declination as sd
from .solar_time_datetime import solar_time_datetime as stda
from .solar_time_hour import solar_time_hour as sth
from .sun_position_collection import sun_position_collection
from .wet_bulb_temperature import wet_bulb_temperature as wbt


def to_dataframe(
    epw: EPW,
) -> pd.DataFrame:
    """Create a Pandas DataFrame from an EPW object, including additional calculated properties.

    Args:
        epw (EPW): An EPW object.

    Returns:
        pd.DataFrame: A Pandas DataFrame with the EPW data and additional calculated properties.
    """

    all_series = []
    for p in dir(epw):
        try:
            all_series.append(to_series(getattr(epw, p)))
        except (AttributeError, TypeError, ZeroDivisionError, ValueError):
            pass

    for k, v in epw.monthly_ground_temperature.items():
        hourly_collection = to_hourly(v)
        hourly_series = to_series(hourly_collection)
        hourly_series.name = f"{hourly_series.name} at {k}m"
        all_series.append(hourly_series)

    # Calculate additional solar properties
    sun_position = sun_position_collection(epw)
    equation_of_time = eot(epw)
    solar_time_hour = sth(epw, equation_of_time)
    solar_altitude = sa(epw, sun_position)
    solar_altitude_in_radians = sar(epw, sun_position)
    solar_declination = sd(epw)
    solar_time_datetime = stda(epw, solar_time_hour)
    solar_azimuth = saz(epw, sun_position)
    solar_azimuth_in_radians = sazr(epw, sun_position)
    apparent_solar_zenith = asz(epw, solar_altitude)
    clearness_index = ci(epw, sun_position)

    # Calculate additional psychrometric properties
    humidity_ratio = hr(epw)
    enthalpy = ent(epw, humidity_ratio)
    wet_bulb_temperature = wbt(epw)

    # Add properties to DataFrame
    for collection in [
        equation_of_time,
        solar_time_hour,
        solar_altitude,
        solar_declination,
        solar_time_datetime,
        solar_azimuth,
        solar_azimuth_in_radians,
        apparent_solar_zenith,
        solar_altitude,
        solar_altitude_in_radians,
        humidity_ratio,
        enthalpy,
        wet_bulb_temperature,
        clearness_index,
    ]:
        all_series.append(to_series(collection))

    # Compile all the data into a dataframe
    df = pd.concat(all_series, axis=1).sort_index(axis=1)

    return df