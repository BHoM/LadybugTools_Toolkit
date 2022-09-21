import warnings
from typing import Dict

import pandas as pd
from ladybug.epw import EPW, MonthlyCollection
from ladybug.location import Location
from ladybugtools_toolkit.ladybug_extension.header.to_string import (
    to_string as header_to_string,
)


from ladybugtools_toolkit import analytics


@analytics
def from_dataframe(
    dataframe: pd.DataFrame,
    location: Location = None,
    monthly_ground_temperature: Dict[float, MonthlyCollection] = None,
    comments_1: str = None,
    comments_2: str = None,
) -> EPW:
    """Create an EPW object from a Pandas DataFrame with named columns.

    Args:
        dataframe (pd.DataFrame):
            A Pandas DataFrame with named columns.
        location (Location, optional):
            A ladybug Location object. Defaults to None which results in a default being applied.
        monthly_ground_temperature (Dict[float, MonthlyCollection], optional):
            A dictionary of monthly ground temperatures. Default is None.
        comments_1 (str, optional):
            A string to be added as comment to the resultant object. Default is None.
        comments_2 (str, optional):
            Another string to be added as comment to the resultant object. Default is None.

    Returns:
        EPW:
            An EPW object.
    """

    # Check dataframe shape for leaped-ness and length
    if sum((dataframe.index.month == 2) & (dataframe.index.day == 29)) != 0:
        leap_yr = True
        if len(dataframe.index) != 8784:
            raise ValueError(
                "The dataframe must have 8784 rows as it contains a 29th of February, suggesting a leap year."
            )
    else:
        leap_yr = False
        if len(dataframe.index) != 8760:
            raise ValueError(
                "The dataframe must have 8760 rows as it does not contain a 29th of February, suggesting a non-leap year."
            )

    # create "empty" EPW object
    epw_obj = EPW.from_missing_values(is_leap_year=leap_yr)

    # Add "location" attributes
    if location is None:
        location = Location()
    location = location.__copy__()
    try:
        location.source += "[Custom EPW from Pandas DataFrame]"
    except TypeError:
        location.source = "[Custom EPW from Pandas DataFrame]"
    epw_obj.location = location

    # Add ground temperatures if available
    if monthly_ground_temperature:
        epw_obj.monthly_ground_temperature = monthly_ground_temperature

    # Add comments if provided
    if comments_1:
        epw_obj.comments_1 = comments_1
    if comments_2:
        epw_obj.comments_2 = comments_2

    # Assign data to the EPW
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

    try:
        for prop in properties:
            setattr(
                getattr(epw_obj, prop),
                "values",
                dataframe[header_to_string(getattr(epw_obj, prop).header)].values,
            )
    except KeyError:
        warnings.warn(
            f"{prop} cannot be added to EPW as it doesn't exist in the Pandas DataFrame."
        )

    return epw_obj
