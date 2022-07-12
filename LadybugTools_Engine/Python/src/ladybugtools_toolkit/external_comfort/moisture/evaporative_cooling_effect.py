from typing import List

from ladybug.psychrometrics import wet_bulb_from_db_rh


def evaporative_cooling_effect(
    dry_bulb_temperature: float,
    relative_humidity: float,
    evaporative_cooling_effectiveness: float,
    atmospheric_pressure: float = None,
) -> List[float]:
    """For the inputs, calculate the effective DBT and RH values for the evaporative cooling effectiveness given.

    Args:
        dry_bulb_temperature (float): A dry bulb temperature in degrees Celsius.
        relative_humidity (float): A relative humidity in percent (0-100).
        evaporative_cooling_effectiveness (float): The evaporative cooling effectiveness. Defaults to 0.3.
        atmospheric_pressure (float, optional): A pressure in Pa.

    Returns:
        List[float]: A list of two values for the effective dry bulb temperature and relative humidity.
    """
    wet_bulb_temperature = wet_bulb_from_db_rh(
        dry_bulb_temperature, relative_humidity, atmospheric_pressure
    )

    return [
        dry_bulb_temperature
        - (
            (dry_bulb_temperature - wet_bulb_temperature)
            * evaporative_cooling_effectiveness
        ),
        (relative_humidity * (1 - evaporative_cooling_effectiveness))
        + evaporative_cooling_effectiveness * 100,
    ]
