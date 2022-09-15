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
def wd_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the wind direction from the simulation directory using a the EPW file to
        assign hourly WD to each point in the simulation, and create the H5 file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial wind direction.
    """

    metric = SpatialMetric.WD_EPW

    wd_path = spatial_metric_filepath(simulation_directory, metric)

    if wd_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        wd_df = pd.read_parquet(wd_path)
        wd_df.columns = wd_df.columns.astype(int)
        return wd_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.index)

    wd_series = to_series(epw.wind_direction)
    wd_df = pd.DataFrame(np.tile(wd_series.values, (n_pts, 1)).T, index=wd_series.index)
    wd_df.columns = wd_df.columns.astype(str)
    wd_df.to_parquet(wd_path)

    return wd_df
