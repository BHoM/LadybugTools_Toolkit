def cardinality(angle_from_north: float, directions: int = 16):
    """Returns the cardinal orientation of a given angle, where that angle is related to north at
        0 degrees.
    Args:
        angle_from_north (float):
            The angle to north in degrees (+Ve is interpreted as clockwise from north at 0.0
            degrees).
        directions (int):
            The number of cardinal directions into which angles shall be binned (This value should
            be one of 4, 8, 16 or 32, and is centred about "north").
    Returns:
        int:
            The cardinal direction the angle represents.
    """

    if angle_from_north > 360 or angle_from_north < 0:
        raise ValueError(
            "The angle entered is beyond the normally expected range for an orientation in degrees."
        )

    cardinal_directions = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        32: [
            "N",
            "NbE",
            "NNE",
            "NEbN",
            "NE",
            "NEbE",
            "ENE",
            "EbN",
            "E",
            "EbS",
            "ESE",
            "SEbE",
            "SE",
            "SEbS",
            "SSE",
            "SbE",
            "S",
            "SbW",
            "SSW",
            "SWbS",
            "SW",
            "SWbW",
            "WSW",
            "WbS",
            "W",
            "WbN",
            "WNW",
            "NWbW",
            "NW",
            "NWbN",
            "NNW",
            "NbW",
        ],
    }

    if directions not in cardinal_directions:
        raise ValueError(
            f'The input "directions" must be one of {list(cardinal_directions.keys())}.'
        )

    val = int((angle_from_north / (360 / directions)) + 0.5)

    arr = cardinal_directions[directions]

    return arr[(val % directions)]
