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


from ladybugtools_toolkit import analytics


@analytics
def rh_evap(simulation_directory: Path, epw: EPW) -> List[pd.DataFrame]:
    """Return the relative humidity from the simulation directory calculating the effective
        RH values following addition of moisture into the air, then create the H5 file to store
        this as a compressed object if not already done.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame:
            A dataframe with the spatial relative-humidity.
    """

    metric = SpatialMetric.RH_EVAP

    rh_matrix_path = spatial_metric_filepath(simulation_directory, metric)
    if rh_matrix_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        return pd.read_parquet(rh_matrix_path)

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    return dbt_rh_evap(simulation_directory, epw)[1]
