from typing import List
from warnings import warn

import numpy as np
from ladybug.location import Location


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


def average_longitude(longitudes: List[float], weights: List[float] = None):
    """Calculate the average longitude from a list of longitudes.

    Args:
        longitudes (List[float]): A list of longitudes.
        weights (List[float], optional): A list of weights for each longitude. Defaults to None which evenly weights each longitude.

    Returns:
        float: The average longitude.
    """

    warn("This method doesnt work for longitudes that cross the -180/180 line...yet")

    longitudes = np.array(longitudes)

    if len(longitudes) == 1:
        return longitudes[0]
    if len(longitudes.shape) != 1:
        raise ValueError("This method only accepts flat lists of longitudes.")
    if weights is None:
        weights = [1 / len(longitudes)] * len(longitudes)
    if len(weights) != len(longitudes):
        raise ValueError("weights must be the same length and longitudes")
    if sum(weights) != 1:
        raise ValueError("weights must sum to 1")

    return np.average(longitudes, weights=weights)


def average_latitude(latitudes: List[float], weights: List[float] = None):
    """Calculate the average latitude from a list of latitudes.

    Args:
        latitudes (List[float]): A list of latitudes.
        weights (List[float], optional): A list of weights for each latitude. Defaults to None which evenly weights each latitude.

    Returns:
        float: The average latitude.
    """

    warn("This method doesnt work for latitudes that cross the -90/90 line...yet")

    latitudes = np.array(latitudes)

    if len(latitudes) == 1:
        return latitudes[0]
    if len(latitudes.shape) != 1:
        raise ValueError("This method only accepts flat lists of latitudes.")
    if weights is None:
        weights = [1 / len(latitudes)] * len(latitudes)
    if len(weights) != len(latitudes):
        raise ValueError("weights must be the same length and latitudes")
    if sum(weights) != 1:
        raise ValueError("weights must sum to 1")

    return np.average(latitudes, weights=weights)


def average_altitude(altitudes: List[float], weights: List[float] = None):
    """Calculate the average altitude from a list of altitudes.

    Args:
        altitudes (List[float]): A list of altitudes.
        weights (List[float], optional): A list of weights for each altitude. Defaults to None which evenly weights each altitude.

    Returns:
        float: The average altitude.
    """

    altitudes = np.array(altitudes)

    if len(altitudes) == 1:
        return altitudes[0]
    if len(altitudes.shape) != 1:
        raise ValueError("This method only accepts flat lists of altitudes.")
    if weights is None:
        weights = [1 / len(altitudes)] * len(altitudes)
    if len(weights) != len(altitudes):
        raise ValueError("weights must be the same length and altitudes")
    if sum(weights) != 1:
        raise ValueError("weights must sum to 1")

    return np.average(altitudes, weights=weights)


def average_location(
    locations: List[Location], weights: List[float] = None
) -> Location:
    """Create an average location from a list of locations. This will use weighting if provided to adjust latitude/longitude values.

    Args:
        locations (List[Location]): A set of ladybug Location objects.
        weights (List[float], optional): A list of weights for each location. Defaults to None which evenly weights each location.

    Returns:
        Location: A synthetic location that is the average of all locations.
    """

    if len(locations) == 1:
        return locations[0]

    if weights is None:
        weights = [1 / len(locations)] * len(locations)

    lat = average_latitude([loc.latitude for loc in locations], weights)
    lon = average_longitude([loc.longitude for loc in locations], weights)
    elv = average_altitude([loc.elevation for loc in locations], weights)

    state = "|".join(
        [
            loc.state if not loc.state in ["", "-", None] else "other"
            for loc in locations
        ]
    )
    city = "|".join(
        [loc.city if not loc.city in ["", "-", None] else "other" for loc in locations]
    )
    country = "|".join(
        [
            loc.country if not loc.country in ["", "-", None] else "other"
            for loc in locations
        ]
    )
    station_id = "|".join(
        [
            loc.station_id if not loc.station_id in ["", "-", None] else "other"
            for loc in locations
        ]
    )
    source = "|".join(
        [
            loc.source if not loc.source in ["", "-", None] else "other"
            for loc in locations
        ]
    )
    return Location(city, state, country, lat, lon, elevation=elv)
