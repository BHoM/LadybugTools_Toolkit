from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect import (
    evaporative_cooling_effect,
)
from ladybugtools_toolkit.external_comfort.spatial.load.evap_clg_magnitude import (
    evap_clg_magnitude,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


from python_toolkit.bhom.analytics import analytics


@analytics
def dbt_rh_evap(simulation_directory: Path, epw: EPW) -> List[pd.DataFrame]:
    """Calculate the effective DBT/RH from a spatially distributed set of moisture sources.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        List[pd.DataFrame]: A list of effective DBT and effective RH.

    """

    dbt_matrix_path = spatial_metric_filepath(
        simulation_directory, SpatialMetric.DBT_EVAP
    )
    rh_matrix_path = spatial_metric_filepath(
        simulation_directory, SpatialMetric.RH_EVAP
    )

    if dbt_matrix_path.exists() and rh_matrix_path.exists():
        dbt_df, rh_df = pd.read_parquet(dbt_matrix_path), pd.read_parquet(
            rh_matrix_path
        )
        dbt_df.columns = dbt_df.columns.astype(int)
        rh_df.columns = rh_df.columns.astype(int)
        return dbt_df, rh_df

    # load evaporative cooling moisture magnitude matrix
    moisture_matrix = evap_clg_magnitude(simulation_directory, epw)

    # create matrix of dbt, rh based on moisture matrix
    # this calculates both dbt and rh, as the both are linked and should be calculated in parallel
    dbt_matrix = []
    rh_matrix = []
    for n, (dt, row) in enumerate(moisture_matrix.iterrows()):
        print(
            f"[{simulation_directory.name}] - Calculating evaporatively cooled DBT/RH ({n / 8760:0.1%})",
            end="\r",
        )
        dbt_base = epw.dry_bulb_temperature[n]
        rh_base = epw.relative_humidity[n]
        atm_base = epw.atmospheric_station_pressure[n]
        if row.sum() == 0:
            dbt_matrix.append(np.tile(dbt_base, len(row)))
            rh_matrix.append(np.tile(rh_base, len(row)))
        else:
            dbt, rh = evaporative_cooling_effect(dbt_base, rh_base, row, atm_base)
            dbt_matrix.append(dbt)
            rh_matrix.append(rh)

    idx = to_series(epw.dry_bulb_temperature).index
    dbt_df = pd.DataFrame(np.array(dbt_matrix), index=idx)
    dbt_df.columns = dbt_df.columns.astype(str)
    dbt_df.to_parquet(dbt_matrix_path)

    rh_df = pd.DataFrame(np.array(rh_matrix), index=idx)
    rh_df.columns = rh_df.columns.astype(str)
    rh_df.to_parquet(rh_matrix_path)

    return dbt_df, rh_df
