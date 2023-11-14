"""Methods for manipulating Ladybug Location objects."""
import numpy as np
from ladybug.location import Location

from ..bhom.analytics import bhom_analytics


@bhom_analytics()
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


@bhom_analytics()
def average_location(
    locations: list[Location], weights: list[float] = None
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

    if len(locations) == 1:
        return locations[0]

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

    state = "|".join(
        [
            loc.state if not loc.state in ["", "-", None] else "NoState"
            for loc in locations
        ]
    )
    city = "|".join(
        [loc.city if not loc.city in ["", "-", None] else "NoCity" for loc in locations]
    )
    country = "|".join(
        [
            loc.country if not loc.country in ["", "-", None] else "NoCountry"
            for loc in locations
        ]
    )
    station_id = "|".join(
        [
            loc.station_id if not loc.station_id in ["", "-", None] else "NoStationId"
            for loc in locations
        ]
    )
    source = "|".join(
        [
            loc.source if not loc.source in ["", "-", None] else "NoSource"
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
