from warnings import warn

from ladybug.datacollection import MonthlyCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.ground_temperature.ground_temperature_at_depth import (
    ground_temperature_at_depth,
)


def monthly_ground_temperature(
    epw: EPW, depth: float = 0.5, soil_diffusivity: float = 0.31e-6
) -> MonthlyCollection:
    """Return the monthly ground temperature values from the EPW file, or approximate these if not available.

    Args:
        epw (EPW):
            The EPW file to extract ground temperature data from.
        depth (float, optional):
            The depth of the soil in meters. Defaults to 0.5 meters.
        soil_diffusivity (float, optional):
            The soil diffusivity in m2/s. Defaults to 0.31e-6 m2/s.

    Returns:
        MonthlyCollection:
            A data collection containing ground temperature values at the depth specified.
    """

    try:
        return epw.monthly_ground_temperature[depth]
    except KeyError:
        warn(
            f"The input EPW doesn't contain any monthly ground temperatures at {depth}m. An "
            "approximation method will be used to determine ground temperature at that depth "
            "based on ambient dry-bulb."
        )
        return ground_temperature_at_depth(
            epw, depth, soil_diffusivity
        ).average_monthly()
