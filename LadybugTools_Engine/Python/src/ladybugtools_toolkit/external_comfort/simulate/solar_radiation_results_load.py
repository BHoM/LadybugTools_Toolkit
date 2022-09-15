from pathlib import Path
from typing import Dict

import pandas as pd
from ladybug.epw import HourlyContinuousCollection
from ladybugtools_toolkit.honeybee_extension.results.load_ill import load_ill
from ladybugtools_toolkit.honeybee_extension.results.make_annual import make_annual
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)


from python_toolkit.bhom.analytics import analytics


@analytics
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
    _shaded = pd.Series(
        [0] * len(unshaded_total), index=unshaded_total.index, name="base"
    )

    return {
        "unshaded_direct_radiation": from_series(unshaded_direct),
        "unshaded_diffuse_radiation": from_series(unshaded_diffuse),
        "unshaded_total_radiation": from_series(unshaded_total),
        "shaded_direct_radiation": from_series(
            _shaded.rename("DirectNormalRadiation (Wh/m2)")
        ),
        "shaded_diffuse_radiation": from_series(
            _shaded.rename("DiffuseHorizontalRadiation (Wh/m2)")
        ),
        "shaded_total_radiation": from_series(
            _shaded.rename("GlobalHorizontalRadiation (Wh/m2)")
        ),
    }
