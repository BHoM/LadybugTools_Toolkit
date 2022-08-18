import warnings


def temperature_at_height(
    reference_temperature: float, reference_height: float, target_height: float
) -> float:
    """Estimate the dry-bulb temperature at a given height from a referenced dry-bulb temperature at another height.

    Args:
        reference_temperature (float):
            The temperature to translate.
        reference_height (float):
            The height of the reference temperature.
        target_height (float):
            The height to translate the reference temperature towards.

    Returns:
        float:
            A translated air temperature.
    """

    if (target_height > 8000) or (reference_height > 8000):
        warnings.warn(
            "The heights input into this calculation exist partially above the egde of the troposphere. This method is only valid below 8000m."
        )

    height_difference = target_height - reference_height

    # rule of thumb - temperature reduces by 6.5C every 1000m in elevation up to troposphere
    return reference_temperature - height_difference / 1000 * 6.5
