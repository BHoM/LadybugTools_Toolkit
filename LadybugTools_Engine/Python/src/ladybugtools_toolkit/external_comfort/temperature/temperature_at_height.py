import warnings


from python_toolkit.bhom.analytics import analytics


@analytics
def temperature_at_height(
    reference_temperature: float,
    reference_height: float,
    target_height: float,
    reduction_per_km_altitude_gain: float = 9.8,
) -> float:
    """Estimate the dry-bulb temperature at a given height from a referenced dry-bulb temperature
        at another height.

    Args:
        reference_temperature (float):
            The temperature to translate.
        reference_height (float):
            The height of the reference temperature.
        target_height (float):
            The height to translate the reference temperature towards.
        reduction_per_km_altitude_gain (float):
            The degrees C reduction for every 1000m altitude gain. Default is 9.8C for clear
            conditions. This would be 6.5C if in cloudy/moist air conditions.

    Returns:
        float:
            A translated air temperature.
    """

    if (target_height > 8000) or (reference_height > 8000):
        warnings.warn(
            "The heights input into this calculation exist partially above the egde of the troposphere. This method is only valid below 8000m."
        )

    height_difference = target_height - reference_height

    return (
        reference_temperature
        - height_difference / 1000 * reduction_per_km_altitude_gain
    )
