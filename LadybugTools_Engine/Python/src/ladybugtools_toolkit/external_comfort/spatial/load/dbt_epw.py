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


def dbt_epw(
    simulation_directory: Path,
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the dry-bulb-temperatures from the simulation directory using an EPW file to
        assign hourly DBT to each point in the simulation, and create the H5 file to store this as
        a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial dry-bulb-temperature.
    """
    metric = SpatialMetric.DBT_EPW

    dbt_path = spatial_metric_filepath(simulation_directory, metric)

    if dbt_path.exists():
        print(f"- Loading {metric.value} from {simulation_directory.name}")
        return pd.read_hdf(dbt_path, "df")

    print(f"- Generating {metric.value} for {simulation_directory.name}")
    spatial_points = points(simulation_directory)
    n_pts = len(spatial_points.columns)

    dbt_series = to_series(epw.dry_bulb_temperature)
    dbt_df = pd.DataFrame(
        np.tile(dbt_series.values, (n_pts, 1)).T, index=dbt_series.index
    )

    dbt_df.to_hdf(dbt_path, "df", complevel=9, complib="blosc")

    return dbt_df
