import warnings

import numpy as np
import pandas as pd
from ladybug import datatype
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.location import Location
from ladybug.psychrometrics import rel_humid_from_db_dpt
from ladybug.sunpath import Sunpath
from ladybug.wea import Wea
from ladybugtools_toolkit.ladybug_extension.header.to_string import (
    to_string as header_to_string,
)


def from_dataframe(dataframe: pd.DataFrame, location: Location = None) -> EPW:
    """Create an EPW object from a Pandas DataFrame with named columns.

    Args:
        dataframe (pd.DataFrame):
            A Pandas DataFrame with named columns.
        location (Location, optional):
            A ladybug Location object. Defaults to None.

    Returns:
        EPW:
            An EPW object.
    """

    # Check dataframe shape for leaped-ness and length
    if dataframe.index.is_leap_year.any():
        leap_yr = True
        assert (
            len(dataframe.index) == 8784
        ), "The dataframe must have 8784 rows for leap years."
    else:
        leap_yr = False
        assert (
            len(dataframe.index) == 8760
        ), "The dataframe must have 8760 rows for non-leap years."

    if location is None:
        location = Location()
    try:
        location.source += "[Custom EPW from Pandas DataFrame]"
    except TypeError:
        location.source = "[Custom EPW from Pandas DataFrame]"

    epw_obj = EPW.from_missing_values(is_leap_year=leap_yr)
    epw_obj.location = location

    # Assign data to the EPW
    _attributes = [
        "aerosol_optical_depth",
        "albedo",
        "atmospheric_station_pressure",
        "ceiling_height",
        "extraterrestrial_direct_normal_radiation",
        "extraterrestrial_horizontal_radiation",
        "liquid_precipitation_depth",
        "liquid_precipitation_quantity",
        "days_since_last_snowfall",
        "dry_bulb_temperature",
        "dew_point_temperature",
        "wind_speed",
        "wind_direction",
        "direct_normal_radiation",
        "snow_depth",
        "diffuse_horizontal_radiation",
        "horizontal_infrared_radiation_intensity",
        "direct_normal_illuminance",
        "diffuse_horizontal_illuminance",
        "precipitable_water",
        "present_weather_codes",
        "present_weather_observation",
        "total_sky_cover",
        "opaque_sky_cover",
        "visibility",
        "zenith_luminance",
    ]
    try:
        for attribute in _attributes:
            setattr(
                getattr(epw_obj, attribute),
                "values",
                dataframe[header_to_string(getattr(epw_obj, attribute).header)].values,
            )
    except KeyError:
        warnings.warn(
            f"{attribute} cannot be added to EPW as it doesn't exist in the Pandas DataFrame."
        )

    try:
        epw_obj.relative_humidity.values = dataframe["Relative Humidity (%)"].values
    except KeyError:
        warnings.warn(
            f"relative_humidity doesn't exist in the Pandas DataFrame, but is being calculated from DBT and DPT."
        )
        epw_obj.relative_humidity.values = (
            HourlyContinuousCollection.compute_function_aligned(
                rel_humid_from_db_dpt,
                [epw_obj.dry_bulb_temperature, epw_obj.dew_point_temperature],
                datatype.fraction.RelativeHumidity(),
                "%",
            ).values
        )

    try:
        epw_obj.global_horizontal_radiation.values = dataframe[
            "Global Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        warnings.warn(
            f"global_horizontal_radiation doesn't exist in the Pandas DataFrame, but is being calculated from DNR and DHR."
        )
        wea = Wea(
            location,
            epw_obj.direct_normal_radiation,
            epw_obj.diffuse_horizontal_radiation,
        )
        epw_obj.global_horizontal_radiation.values = (
            wea.global_horizontal_irradiance.values
        )

    try:
        epw_obj.global_horizontal_illuminance.values = dataframe[
            "Global Horizontal Illuminance (lux)"
        ].values
    except KeyError:
        warnings.warn(
            f"global_horizontal_illuminance doesn't exist in the Pandas DataFrame, but is being calculated from DNI and DHI."
        )
        glob_horiz = []
        sp = Sunpath.from_location(location)
        sp.is_leap_year = leap_yr
        for dt, dni, dhi in zip(
            epw_obj.direct_normal_illuminance.datetimes,
            epw_obj.direct_normal_illuminance,
            epw_obj.diffuse_horizontal_illuminance,
        ):
            sun = sp.calculate_sun_from_date_time(dt)
            glob_horiz.append(dhi + dni * np.sin(np.radians(sun.altitude)))
        epw_obj.global_horizontal_illuminance.values = glob_horiz

    return epw_obj
