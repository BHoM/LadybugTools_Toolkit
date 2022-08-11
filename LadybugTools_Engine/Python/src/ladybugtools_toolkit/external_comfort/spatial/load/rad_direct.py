from pathlib import Path

import pandas as pd
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.honeybee_extension.results.load_ill import load_ill
from ladybugtools_toolkit.honeybee_extension.results.make_annual import make_annual


def rad_direct(simulation_directory: Path) -> pd.DataFrame:
    """Return the direct irradiance from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a direct irradiance ILL file.

    Returns:
        pd.DataFrame:
            A dataframe with the direct irradiance.
    """

    metric = SpatialMetric.RAD_DIRECT

    rad_direct_path = spatial_metric_filepath(simulation_directory, metric)

    if rad_direct_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        rad_direct_df = pd.read_parquet(rad_direct_path)
        rad_direct_df.columns = rad_direct_df.columns.astype(int)
        return rad_direct_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    ill_files = list(
        (simulation_directory / "annual_irradiance" / "results" / "direct").glob(
            "*.ill"
        )
    )
    rad_direct_df = make_annual(load_ill(ill_files)).fillna(0)

    rad_direct_df = rad_direct_df.clip(lower=0).droplevel(0, axis=1)
    rad_direct_df.columns = rad_direct_df.columns.astype(str)
    rad_direct_df.to_parquet(rad_direct_path)

    return rad_direct_df
