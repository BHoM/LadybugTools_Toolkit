from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.ground_temperature.ground_temperature_at_depth import (
    ground_temperature_at_depth,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.monthlycollection.to_hourly import (
    to_hourly,
)


from ladybugtools_toolkit import analytics


@analytics
def hourly_ground_temperature(
    epw: EPW, depth: float = 0.5, soil_diffusivity: float = 0.31e-6
) -> HourlyContinuousCollection:
    """Return the hourly ground temperature values from the EPW file (upsampled from any
        available monthly values), or approximate these if not available.

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
        return to_hourly(epw.monthly_ground_temperature[depth])
    except KeyError:
        return ground_temperature_at_depth(epw, depth, soil_diffusivity)
