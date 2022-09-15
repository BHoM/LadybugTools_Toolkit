import warnings

import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.ladybug_extension.epw.clearness_index import (
    clearness_index as ci,
)
from ladybugtools_toolkit.ladybug_extension.epw.enthalpy import enthalpy as ent
from ladybugtools_toolkit.ladybug_extension.epw.equation_of_time import (
    equation_of_time as eot,
)
from ladybugtools_toolkit.ladybug_extension.epw.humidity_ratio import (
    humidity_ratio as hr,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude import (
    solar_altitude as sa,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_altitude_radians import (
    solar_altitude_radians as sar,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_azimuth import (
    solar_azimuth as saz,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_azimuth_radians import (
    solar_azimuth_radians as sazr,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_declination import (
    solar_declination as sd,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_time_datetime import (
    solar_time_datetime as stda,
)
from ladybugtools_toolkit.ladybug_extension.epw.solar_time_hour import (
    solar_time_hour as sth,
)
from ladybugtools_toolkit.ladybug_extension.epw.sun_position_collection import (
    sun_position_collection,
)
from ladybugtools_toolkit.ladybug_extension.epw.wet_bulb_temperature import (
    wet_bulb_temperature as wbt,
)

from python_toolkit.bhom.analytics import analytics


@analytics
def to_dataframe(epw: EPW, include_additional: bool = True) -> pd.DataFrame:
    """Create a Pandas DataFrame from an EPW object, with option for including additional metrics.

    Args:
        epw (EPW):
            An EPW object.
        include_additional (bool, optional):
            Set to False to not include additional calculated properties. Default is True.

    Returns:
        pd.DataFrame:
            A Pandas DataFrame containing the source EPW data.
    """

    properties = [
        "aerosol_optical_depth",
        "albedo",
        "atmospheric_station_pressure",
        "ceiling_height",
        "days_since_last_snowfall",
        "dew_point_temperature",
        "diffuse_horizontal_illuminance",
        "diffuse_horizontal_radiation",
        "direct_normal_illuminance",
        "direct_normal_radiation",
        "dry_bulb_temperature",
        "extraterrestrial_direct_normal_radiation",
        "extraterrestrial_horizontal_radiation",
        "global_horizontal_illuminance",
        "global_horizontal_radiation",
        "horizontal_infrared_radiation_intensity",
        "liquid_precipitation_depth",
        "liquid_precipitation_quantity",
        "opaque_sky_cover",
        "precipitable_water",
        "present_weather_codes",
        "present_weather_observation",
        "relative_humidity",
        "snow_depth",
        "total_sky_cover",
        "visibility",
        "wind_direction",
        "wind_speed",
        "years",
        "zenith_luminance",
    ]

    all_series = []
    for prop in properties:
        try:
            all_series.append(to_series(getattr(epw, prop)))
        except ValueError:
            warnings.warn(
                f"{prop} is not available in this EPW file. This is most likely because this file does not follow normal EPW content conventions."
            )

    if not include_additional:
        return pd.concat(all_series, axis=1).sort_index(axis=1)

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
        solar_altitude,
        solar_altitude_in_radians,
        humidity_ratio,
        enthalpy,
        wet_bulb_temperature,
        clearness_index,
    ]:
        all_series.append(to_series(collection))

    return pd.concat(all_series, axis=1).sort_index(axis=1)
