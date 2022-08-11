from pathlib import Path

import pandas as pd
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.honeybee_extension.results.load_pts import load_pts


def points(simulation_directory: Path) -> pd.DataFrame:
    """Return the points results from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a pts file.

    Returns:
        pd.DataFrame:
            A dataframe with the points locations.
    """

    metric = SpatialMetric.POINTS
    points_path = spatial_metric_filepath(simulation_directory, metric)

    if points_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        return pd.read_parquet(points_path)

    print(f"[{simulation_directory.name}] - Generating {metric.value}")
    points_files = list(
        (simulation_directory / "sky_view" / "model" / "grid").glob("*.pts")
    )
    points_df = load_pts(points_files).droplevel(0, axis=1)
    points_df.to_parquet(points_path)
    return points_df
