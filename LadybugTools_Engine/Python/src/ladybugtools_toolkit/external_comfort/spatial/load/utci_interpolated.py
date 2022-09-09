from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import \
    SpatialMetric
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import \
    spatial_metric_filepath
from ladybugtools_toolkit.external_comfort.spatial.unshaded_shaded_interpolation import \
    unshaded_shaded_interpolation


def utci_interpolated(
    simulation_directory: Path,
    unshaded_universal_thermal_climate_index: HourlyContinuousCollection,
    shaded_universal_thermal_climate_index: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view: pd.DataFrame,
    epw: EPW,
) -> pd.DataFrame:
    """Return the UTCI from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Path):
            The simulation directory.
        unshaded_universal_thermal_climate_index (HourlyContinuousCollection):
            A collection containing unshaded UTCI values.
        shaded_universal_thermal_climate_index (HourlyContinuousCollection):
            A collection containing shaded UTCI values.
        total_irradiance (pd.DataFrame):
            A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame):
            A dataframe containing point-wise sky-view values.
        epw (EPW):
            The associate EPW file.

    Returns:
        pd.DataFrame:
            A dataframe containing spatial UTCI values.
    """

    metric = SpatialMetric.UTCI_INTERPOLATED

    utci_path = spatial_metric_filepath(simulation_directory, metric)

    if utci_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        utci_df = pd.read_parquet(utci_path)
        utci_df.columns = utci_df.columns.astype(int)
        return utci_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    utci_df = unshaded_shaded_interpolation(
        unshaded_universal_thermal_climate_index,
        shaded_universal_thermal_climate_index,
        total_irradiance,
        sky_view,
        np.array(epw.global_horizontal_radiation.values) > 0,
    )
    utci_df.columns = utci_df.columns.astype(str)

    utci_df.to_parquet(utci_path)
    return utci_df
