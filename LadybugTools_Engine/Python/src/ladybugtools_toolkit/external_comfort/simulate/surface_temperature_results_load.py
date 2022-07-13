from pathlib import Path
from typing import Dict

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW

from ...honeybee_extension.results import load_sql
from ...ladybug_extension.datacollection import from_series


def surface_temperature_results_load(
    sql_path: Path,
    epw: EPW,
) -> Dict[str, HourlyContinuousCollection]:
    """Load results from the surface temperature simulation.

    Args:
        sql_path (Path): An SQL file containing EnergyPlus results.
        epw (EPW): An EPW file. Required to get the temperature of the sky for an unshaded case.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing surface temperature-related collections.
    """

    # Return results
    df = load_sql(sql_path)

    return {
        "shaded_below_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_SHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
        ),
        "unshaded_below_temperature": from_series(
            df.filter(regex="GROUND_ZONE_UP_UNSHADED")
            .droplevel([0, 1, 2], axis=1)
            .squeeze()
        ),
        "shaded_above_temperature": from_series(
            df.filter(regex="SHADE_ZONE_DOWN").droplevel([0, 1, 2], axis=1).squeeze()
        ),
        "unshaded_above_temperature": epw.sky_temperature,
    }
