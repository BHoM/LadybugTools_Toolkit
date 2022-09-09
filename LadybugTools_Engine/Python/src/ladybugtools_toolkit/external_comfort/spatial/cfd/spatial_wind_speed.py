from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.cfd.load_cfd_results import (
    load_cfd_results,
)
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.unique_wind_speed_direction import (
    unique_wind_speed_direction,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from scipy.interpolate import interp1d


def spatial_wind_speed(simulation_directory: Path, epw: EPW) -> pd.DataFrame:
    """Calculate the temporo-spatial wind speed for a given SpatialComfort case.

    Args:
        simulation_directory (Path):
            The associated simulation directory
        epw (EPW):
            An EPW object

    Returns:
        pd.DataFrame:
            A time indexed, spatial matrix of point-wind-speeds.
    """

    # for each unique wind direction, get the interpolated normalised wind speed
    uniques = unique_wind_speed_direction(epw)
    unique_directions = np.unique(uniques.T[1])

    # get the cfd results to interpolate between
    cfd_results = load_cfd_results(simulation_directory)

    # interpolate to unique directions from known directions
    x = cfd_results.columns
    y = cfd_results.values
    z = pd.DataFrame(interp1d(x, y)(unique_directions), columns=unique_directions)

    # for each unique combo of ws and wd, get the associated pt-ws
    d = {}
    for n, (ws, wd) in enumerate(uniques):
        d[(ws, wd)] = (z[wd] * ws).values

    # for each row of epw, create spatial velocity
    vals = []
    for n, (ws, wd) in enumerate(list(zip(*[epw.wind_speed, epw.wind_direction]))):
        print(
            f"[{simulation_directory.stem}] - Calculating spatial wind speed ({n/len(epw.wind_speed):04.1%})",
            end="\r",
        )
        vals.append(d[(ws, wd)])
    idx = to_series(epw.wind_direction).index
    vel_df = pd.DataFrame(np.array(vals), index=idx)

    return vel_df
