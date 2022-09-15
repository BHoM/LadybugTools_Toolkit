from ladybug.location import Location


from python_toolkit.bhom.analytics import analytics


@analytics
def to_string(location: Location) -> str:
    """Return a simple string representation of the Location object.

    Args:
        location (Location):
            A Ladybug location object.

    Returns:
        str:
            A simple string representation of the Location object.
    """
    return f"{location.country} - {location.city}"
