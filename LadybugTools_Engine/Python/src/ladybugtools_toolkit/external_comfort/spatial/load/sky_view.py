from pathlib import Path

import pandas as pd
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.honeybee_extension.results.load_res import load_res


def sky_view(simulation_directory: Path) -> pd.DataFrame:
    """Get the sky view from the simulation directory.

    Args:
        simulation_directory (Path):
            The simulation directory containing a sky view RES file.

    Returns:
        pd.DataFrame:
            The sky view dataframe.
    """

    metric = SpatialMetric.SKY_VIEW

    sky_view_path = spatial_metric_filepath(simulation_directory, metric)

    if sky_view_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        sky_view_df = pd.read_parquet(sky_view_path)
        return sky_view_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    res_files = list((simulation_directory / "sky_view" / "results").glob("*.res"))
    sky_view_df = load_res(res_files).clip(lower=0, upper=100)
    sky_view_df.to_parquet(sky_view_path)

    return sky_view_df
