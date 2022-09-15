from ladybug.epw import EPW


from python_toolkit.bhom.analytics import analytics


@analytics
def equality(epw0: EPW, epw1: EPW, include_header: bool = False) -> bool:
    """Check for equality between two EPW objects, with regards to the data contained within.

    Args:
        epw0 (EPW):
            A ladybug EPW object.
        epw1 (EPW):
            A ladybug EPW object.
        include_header (bool, optional):
            Include the str representation of the EPW files header in the comparison.
            Defaults to False.

    Returns:
        bool:
            True if the two EPW objects are equal, False otherwise.
    """

    if not isinstance(epw0, EPW) or not isinstance(epw1, EPW):
        raise TypeError("Both inputs must be of type EPW.")

    if include_header:
        if epw0.header != epw1.header:
            return False

    # Check key metrics
    for var in [
        "dry_bulb_temperature",
        "relative_humidity",
        "dew_point_temperature",
        "wind_speed",
        "wind_direction",
        "global_horizontal_radiation",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
        "atmospheric_station_pressure",
    ]:
        if getattr(epw0, var) != getattr(epw1, var):
            print(f"{__file__}, {var}")
            return False

    return True
