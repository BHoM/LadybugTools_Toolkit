from typing import List

from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.epw import EPW, HourlyContinuousCollection
from ladybug.psychrometrics import wet_bulb_from_db_rh


def evaporative_cooling_effect(
    dry_bulb_temperature: float,
    relative_humidity: float,
    evaporative_cooling_effectiveness: float,
    atmospheric_pressure: float = None,
) -> List[float]:
    """
    For the inputs, calculate the effective DBT and RH values for the evaporative cooling
    effectiveness given.

    Args:
        dry_bulb_temperature (float):
            A dry bulb temperature in degrees Celsius.
        relative_humidity (float):
            A relative humidity in percent (0-100).
        evaporative_cooling_effectiveness (float):
            The evaporative cooling effectiveness. This should be a value between 0 (no effect)
            and 1 (saturated air).
        atmospheric_pressure (float, optional):
            A pressure in Pa. Default is pressure at sea level (101325 Pa).

    Returns:
        effective_dry_bulb_temperature, effective_relative_humidity (List[float]):
            A list of two values for the effective dry bulb temperature and relative humidity.
    """

    if atmospheric_pressure is None:
        atmospheric_pressure = 101325

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


def evaporative_cooling_effect_collection(
    epw: EPW, evaporative_cooling_effectiveness: float = 0.3
) -> List[HourlyContinuousCollection]:
    """Calculate the effective DBT and RH considering effects of evaporative cooling.

    Args:
        epw (EPW): A ladybug EPW object.
        evaporative_cooling_effectiveness (float, optional):
            The proportion of difference between DBT and WBT by which to adjust DBT.
            Defaults to 0.3 which equates to 30% effective evaporative cooling, roughly that
            of Misting.

    Returns:
        List[HourlyContinuousCollection]:
            Adjusted dry-bulb temperature and relative humidity collections incorporating
            evaporative cooling effect.
    """

    if (evaporative_cooling_effectiveness > 1) or (
        evaporative_cooling_effectiveness < 0
    ):
        raise ValueError("evaporative_cooling_effectiveness must be between 0 and 1.")

    wbt = HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        WetBulbTemperature(),
        "C",
    )
    dbt = epw.dry_bulb_temperature.duplicate()
    dbt = dbt - ((dbt - wbt) * evaporative_cooling_effectiveness)
    dbt.header.metadata[
        "evaporative_cooling"
    ] = f"{evaporative_cooling_effectiveness:0.0%}"

    rh = epw.relative_humidity.duplicate()
    rh = (rh * (1 - evaporative_cooling_effectiveness)) + (
        evaporative_cooling_effectiveness * 100
    )
    rh.header.metadata[
        "evaporative_cooling"
    ] = f"{evaporative_cooling_effectiveness:0.0%}"

    return [dbt, rh]
