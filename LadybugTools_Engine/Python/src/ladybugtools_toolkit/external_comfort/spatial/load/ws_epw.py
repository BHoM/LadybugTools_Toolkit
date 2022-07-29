from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.load.points import points
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def ws_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the wind speed from the simulation directory using a the EPW file to
        assign hourly WS to each point in the simulation, and create the H5 file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial wind speed.
    """

    metric = SpatialMetric.WS_EPW

    ws_path = spatial_metric_filepath(simulation_directory, metric)

    if ws_path.exists():
        print(f"- Loading {metric.value} from {simulation_directory.name}")
        return pd.read_hdf(ws_path, "df")

    print(f"- Generating {metric.value} for {simulation_directory.name}")
    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    ws_series = to_series(epw.wind_speed)
    ws_df = pd.DataFrame(np.tile(ws_series.values, (n_pts, 1)).T, index=ws_series.index)

    ws_df.to_hdf(ws_path, "df", complevel=9, complib="blosc")

    return ws_df
