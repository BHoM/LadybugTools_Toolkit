from pathlib import Path
from typing import List

import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.load.dbt_rh_evap import dbt_rh_evap
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)


def dbt_evap(simulation_directory: Path, epw: EPW) -> List[pd.DataFrame]:
    """Return the dry-bulb-temperatures from the simulation directory calculating the effective
        DBT values following addition of moisture into the air, then create the H5 file to store
        this as a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial dry-bulb-temperature.
    """
    metric = SpatialMetric.DBT_EVAP

    dbt_matrix_path = spatial_metric_filepath(simulation_directory, metric)

    if dbt_matrix_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        return pd.read_hdf(dbt_matrix_path, "df")

    print(f"[{simulation_directory.name}] - Generating {metric.value}")
    return dbt_rh_evap(simulation_directory, epw)[0]
