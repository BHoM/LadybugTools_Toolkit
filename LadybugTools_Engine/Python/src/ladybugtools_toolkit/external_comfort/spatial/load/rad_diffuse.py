from pathlib import Path

import pandas as pd
from ladybugtools_toolkit import analytics
from ladybugtools_toolkit.external_comfort.spatial.load.rad_direct import \
    rad_direct
from ladybugtools_toolkit.external_comfort.spatial.load.rad_total import \
    rad_total
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import \
    SpatialMetric
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import \
    spatial_metric_filepath


@analytics
def rad_diffuse(
    simulation_directory: Path,
    total_irradiance: pd.DataFrame = None,
    direct_irradiance: pd.DataFrame = None,
) -> pd.DataFrame:
    """Return the diffuse irradiance from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The directory containing a diffuse irradiance ILL file.
        total_irradiance (pd.DataFrame, optional):
            If given along with direct_irradiance, then calculation will be completed faster.
            Default is None.
        direct_irradiance (pd.DataFrame, optional):
            If given along with total_irradiance, then calculation will be completed faster.
            Default is None.

    Returns:
        pd.DataFrame: A dataframe with the diffuse irradiance.
    """

    metric = SpatialMetric.RAD_DIFFUSE

    rad_diffuse_path = spatial_metric_filepath(simulation_directory, metric)

    if rad_diffuse_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        return pd.read_parquet(rad_diffuse_path)

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    if (total_irradiance is None) and (direct_irradiance is None):
        total_irradiance = rad_total(simulation_directory)
        direct_irradiance = rad_direct(simulation_directory)

    rad_diffuse_df = total_irradiance - direct_irradiance

    rad_diffuse_df.columns = rad_diffuse_df.columns.astype(str)

    rad_diffuse_df.clip(lower=0).to_parquet(rad_diffuse_path)
    return rad_diffuse_df
