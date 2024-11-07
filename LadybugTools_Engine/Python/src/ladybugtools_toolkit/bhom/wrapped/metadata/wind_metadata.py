from ladybugtools_toolkit.wind import Wind


def wind_metadata(
    wind_object: Wind,
    directions: int = 36,
    ignore_calm: bool = True,
    threshold: float = 1e-10,
) -> dict:
    """Provides a dictionary containing metadata of this wind object.

    Args:
        directions (int, optional):
            The number of directions to use. Defaults to 36.
        ignore_calm (bool, optional):
            Whether or not to ignore wind speed values before the threshold, allowing a more accurate prevailing direction and quantile wind speeds. Defaults to True
        threshold (float, optional):
            The threshold to use for calm wind speeds. Defaults to 1e-10.

    Returns:
        dict:
            The resultant metadata dictionary, which has the following structure:
            {
                "95percentile": 95 percentile wind speed,
                "50percentile": 50 percentile wind speed,
                "calm_percent": the proportion of calm hours (not affected by ignore_calm bool),
                "prevailing_direction": direction of prevailing wind,
                "prevailing_95percentile": prevailing 95 percentile wind speed,
                "prevailing_50percentile": prevailing 50 percentile wind speed
            }

    """
    ws = wind_object.ws

    prevailing_wind_speeds, prevailing_directions = wind_object.prevailing_wind_speeds(
        n=1, directions=directions, ignore_calm=ignore_calm, threshold=threshold)

    prevailing_wind_speed = prevailing_wind_speeds[0]
    prevailing_direction = prevailing_directions[0]

    return {
        "95percentile": ws.quantile(0.95),
        "50percentile": ws.quantile(0.50),
        "calm_percent": wind_object.calm(),
        "prevailing_direction": prevailing_direction,
        "prevailing_95percentile": prevailing_wind_speed.quantile(0.95),
        "prevailing_50percentile": prevailing_wind_speed.quantile(0.5),
    }
