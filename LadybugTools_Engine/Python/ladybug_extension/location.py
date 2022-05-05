from ladybug.location import Location


def describe(location: Location) -> str:
    """Return a simple string representation of the LOcation object.

    Args:
        location (Location): A Ladybug location object.

    Returns:
        str: A simple string representation of the Location object.
    """
    return f"{location.country} - {location.city}"
