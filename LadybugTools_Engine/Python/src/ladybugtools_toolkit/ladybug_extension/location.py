"""Methods for interacting with Ladybug Location objects."""

import warnings
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Union

import geopandas as gpd
import numpy as np
import pytz
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.location import Location

from .. import DATA_DIRECTORY
from ..bhom.logger import CONSOLE_LOGGER
from python_toolkit.bhom.analytics import bhom_analytics

def location_to_string(location: Location) -> str:
    """Return a simple string representation of the Location object.

    Args:
        location (Location):
            A Ladybug location object.

    Returns:
        str:
            A simple string representation of the Location object.
    """
    return f"{location.country.strip()} - {location.city.strip()}"

def great_circle_distance(location1: Location, location2: Location) -> float:
    """Calculate the great circle distance between two points on the earth
    (specified in decimal degrees), in metres.

    Args:
        location1 (Location):
            Location object of the first location
        location2 (Location):
            Location object of the second location

    Returns:
        distance (float):
            The distance between the two locations in m

    """
    r = 6373.0  # approximate radius of earth in km
    lat1 = np.radians(location1.latitude)
    lon1 = np.radians(location1.longitude)
    lat2 = np.radians(location2.latitude)
    lon2 = np.radians(location2.longitude)
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = r * c
    return distance * 1000

def get_tzinfo(latitude: float, longitude: float) -> tzinfo:
    """Get the timezone fixed offset in hours for a given latitude and longitude.
    This function assumes the date is January 1st, to avoid DST issues.
    
    Args:
        latitude:
            The latitude of the location.
        longitude:
            The longitude of the location.

    """
    # load the data
    tz_shapefile = DATA_DIRECTORY / "tz" / "timezones.shp"
    df = gpd.read_file(tz_shapefile)

    # get the timezone from looking up where the point lies on a map of timezones
    tz_offset = None
    for _, row in df.iterrows():
        if row.geometry.contains(gpd.points_from_xy([longitude], [latitude])[0]):
            tz_offset = row["tzoffset"]
            break

    # if no timezone found, use UTC as default
    if tz_offset is None:
        CONSOLE_LOGGER.warning(
            f"No timezone offset found for latitude {latitude} and longitude {longitude}. "
            "Using 0 as default."
        )
        tz_offset = 0.0

    # create a timezone object with the offset
    return pytz.FixedOffset(tz_offset * 60)


def location_to_timezone(location: Location) -> timezone:
    """Convert a Ladybug Location object to a timezone object."""
    return timezone(offset=timedelta(hours=location.time_zone))


def average_location(
    locations: list[Location], weights: Union[tuple[Union[int, float]], None] = None
) -> Location:
    """Create an average location from a list of locations.
    This will use weighting if provided to adjust latitude/longitude values.

    Args:
        locations (list[Location]):
            A set of ladybug Location objects.
        weights (list[float], optional):
            A list of weights for each location.
            Defaults to None which evenly weights each location.

    Returns:
        Location: A synthetic location that is the average of all locations.

    """
    # check inputs

    if not isinstance(locations, (list, tuple)):
        raise TypeError("Locations must be a list or tuple of Location objects.")

    if len(locations) == 1:
        return locations[0]

    if len(locations) == 0:
        raise ValueError("No locations provided.")

    if weights is None:
        weights = [1] * len(locations)

    if len(weights) != len(locations):
        raise ValueError("The number of weights must match the number of locations.")

    if sum(weights) == 0:
        raise ValueError("The sum of weights cannot be zero.")

    # raise a warning is the locations are quite far away
    distances = []
    for loc1 in locations:
        for loc2 in locations:
            distances.append(great_circle_distance(loc1, loc2))
    if max(distances) > 10000:
        CONSOLE_LOGGER.warning(
            f"The maximum distance between the locations passed is {max(distances)} km. Consider that this combined location may not be useful (for instance when used with the Wind class)"
        )

    # calculate average latitude, longitude, and elevation
    lat = (
        np.average(
            np.array([loc.latitude for loc in locations]) + 1000, weights=weights
        )
        - 1000
    )
    lon = (
        np.average(
            np.array([loc.longitude for loc in locations]) + 1000, weights=weights
        )
        - 1000
    )
    elv = np.average(np.array([loc.elevation for loc in locations]), weights=weights)

    # create the location descriptors
    state = "|".join(
        [
            loc.state if loc.state not in ["", "-", None] else "NoState"
            for loc in locations
        ]
    )
    city = "|".join(
        [loc.city if loc.city not in ["", "-", None] else "NoCity" for loc in locations]
    )
    country = "|".join(
        [
            str(loc.country) if loc.country not in ["", "-", None] else "NoCountry"
            for loc in locations
        ]
    )
    station_id = "|".join(
        [
            str(loc.station_id)
            if loc.station_id not in ["", "-", None]
            else "NoStationId"
            for loc in locations
        ]
    )
    source = "|".join(
        [
            str(loc.source) if loc.source not in ["", "-", None] else "NoSource"
            for loc in locations
        ]
    )
    return Location(
        city=f"Synthetic ({city})",
        state=f"Synthetic ({state})",
        country=f"Synthetic ({country})",
        latitude=lat,
        longitude=lon,
        elevation=elv,
        station_id=f"Synthetic ({station_id})",
        source=f"Synthetic ({source})",
    )


def location_to_pytz_fixed_offset(location: Location) -> pytz._FixedOffset:
    """Convert a ladybug time zone (in hours) to a pytz fixed offset object."""
    if not isinstance(location, Location):
        raise TypeError("lb_time_zone must be an int or float.")
    return pytz.FixedOffset(int(location.time_zone * 60))
