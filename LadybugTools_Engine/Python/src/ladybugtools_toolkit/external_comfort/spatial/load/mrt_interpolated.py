from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.external_comfort.spatial.unshaded_shaded_interpolation import (
    unshaded_shaded_interpolation,
)


from python_toolkit.bhom.analytics import analytics


@analytics
def mrt_interpolated(
    simulation_directory: Path,
    unshaded_mean_radiant_temperature: HourlyContinuousCollection,
    shaded_mean_radiant_temperature: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view: pd.DataFrame,
    epw: EPW,
) -> pd.DataFrame:
    """Return the MRT from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The simulation directory.
        unshaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing unshaded MRT values.
        shaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing shaded MRT values.
        total_irradiance (pd.DataFrame):
            A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame):
            A dataframe containing point-wise sky-view values.
        epw (EPW):
            The associate EPW file.

    Returns:
        pd.DataFrame:
            A dataframe containing spatial MRT values.
    """
    metric = SpatialMetric.MRT_INTERPOLATED
    mrt_path = spatial_metric_filepath(simulation_directory, metric)

    if mrt_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        mrt_df = pd.read_parquet(mrt_path)
        mrt_df.columns = mrt_df.columns.astype(int)
        return mrt_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")
    mrt_df = unshaded_shaded_interpolation(
        unshaded_mean_radiant_temperature,
        shaded_mean_radiant_temperature,
        total_irradiance,
        sky_view,
        np.array(epw.global_horizontal_radiation.values) > 0,
    )
    mrt_df.columns = mrt_df.columns.astype(str)
    mrt_df.to_parquet(mrt_path)

    return mrt_df
