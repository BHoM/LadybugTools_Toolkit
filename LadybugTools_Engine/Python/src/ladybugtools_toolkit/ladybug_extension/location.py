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
