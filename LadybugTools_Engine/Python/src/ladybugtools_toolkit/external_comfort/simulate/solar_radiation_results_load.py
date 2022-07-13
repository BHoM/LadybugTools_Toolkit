from pathlib import Path
from typing import Dict

import pandas as pd
from ladybug.epw import HourlyContinuousCollection

from ...honeybee_extension.results import load_ill, make_annual
from ...ladybug_extension.datacollection import from_series


def solar_radiation_results_load(
    total_irradiance: Path, direct_irradiance: Path
) -> Dict[str, HourlyContinuousCollection]:
    """Load results from the solar radiation simulation.

    Args:
        total_irradiance (Path): An ILL file containing total irradiance.
        direct_irradiance (Path): An ILL file containing direct irradiance.

    Returns:
        Dict[str, HourlyContinuousCollection]: A dictionary containing radiation-related collections.
    """

    unshaded_total = (
        make_annual(load_ill(total_irradiance))
        .fillna(0)
        .sum(axis=1)
        .rename("GlobalHorizontalRadiation (Wh/m2)")
    )
    unshaded_direct = (
        make_annual(load_ill(direct_irradiance))
        .fillna(0)
        .sum(axis=1)
        .rename("DirectNormalRadiation (Wh/m2)")
    )
    unshaded_diffuse = (unshaded_total - unshaded_direct).rename(
        "DiffuseHorizontalRadiation (Wh/m2)"
    )

    return {
        "unshaded_direct_radiation": from_series(unshaded_direct),
        "unshaded_diffuse_radiation": from_series(unshaded_diffuse),
        "shaded_direct_radiation": from_series(
            pd.Series(
                [0] * 8760,
                index=unshaded_total.index,
                name="DirectNormalRadiation (Wh/m2)",
            )
        ),
        "shaded_diffuse_radiation": from_series(
            pd.Series(
                [0] * 8760,
                index=unshaded_total.index,
                name="DiffuseHorizontalRadiation (Wh/m2)",
            )
        ),
    }
