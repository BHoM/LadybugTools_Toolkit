from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.cfd.spatial_wind_speed import (
    spatial_wind_speed,
)
from ladybugtools_toolkit.external_comfort.spatial.load.points import points
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def ws_cfd(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the wind speed from the simulation directory using results from a CFD simulation,
        and create the H5 file to store this as a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial wind speed.
    """

    metric = SpatialMetric.WS_CFD

    ws_path = spatial_metric_filepath(simulation_directory, metric)

    if ws_path.exists():
        print(f"- Loading {metric.value} from {simulation_directory.name}")
        return pd.read_hdf(ws_path, "df")

    print(f"- Generating {metric.value} for {simulation_directory.name}")

    ws_df = spatial_wind_speed(simulation_directory, epw)

    ws_df.to_hdf(ws_path, "df", complevel=9, complib="blosc")

    return ws_df
