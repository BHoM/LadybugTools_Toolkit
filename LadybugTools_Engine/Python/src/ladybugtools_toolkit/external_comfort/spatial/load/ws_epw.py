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


from python_toolkit.bhom.analytics import analytics


@analytics
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
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        ws_df = pd.read_parquet(ws_path)
        ws_df.columns = ws_df.columns.astype(int)
        return ws_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")
    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    ws_series = to_series(epw.wind_speed)
    ws_df = pd.DataFrame(np.tile(ws_series.values, (n_pts, 1)).T, index=ws_series.index)
    ws_df.columns = ws_df.columns.astype(str)
    ws_df.to_parquet(ws_path)

    return ws_df
