from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.load.dbt_epw import dbt_epw
from ladybugtools_toolkit.external_comfort.spatial.load.dbt_evap import dbt_evap
from ladybugtools_toolkit.external_comfort.spatial.load.mrt_interpolated import (
    mrt_interpolated,
)
from ladybugtools_toolkit.external_comfort.spatial.load.rh_epw import rh_epw
from ladybugtools_toolkit.external_comfort.spatial.load.rh_evap import rh_evap
from ladybugtools_toolkit.external_comfort.spatial.load.ws_cfd import ws_cfd
from ladybugtools_toolkit.external_comfort.spatial.load.ws_epw import ws_epw
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.external_comfort.spatial.unshaded_shaded_interpolation import (
    unshaded_shaded_interpolation,
)
from ladybugtools_toolkit.external_comfort.thermal_comfort.utci.utci import utci


def utci_calculated(
    simulation_directory: Path,
    epw: EPW,
    unshaded_mean_radiant_temperature: HourlyContinuousCollection,
    shaded_mean_radiant_temperature: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view: pd.DataFrame,
) -> pd.DataFrame:
    """Return the UTCI calculated using available matrices from moisture effects and/or CFD.

    Args:
        simulation_directory (Path):
            The simulation directory.
        epw (EPW):
            The associate EPW file.
        unshaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing unshaded MRT values.
        shaded_mean_radiant_temperature (HourlyContinuousCollection):
            A collection containing shaded MRT values.
        total_irradiance (pd.DataFrame):
            A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame):
            A dataframe containing point-wise sky-view values.

    Returns:
        pd.DataFrame:
            A dataframe containing spatial UTCI values.
    """

    metric = SpatialMetric.UTCI_CALCULATED

    utci_path = spatial_metric_filepath(simulation_directory, metric)

    if utci_path.exists():
        print(f"- Loading {metric.value} from {simulation_directory.name}")
        return pd.read_hdf(utci_path, "df")

    print(f"- Generating {metric.value} for {simulation_directory.name}")

    # check that moisture impacted RH/DBT is available, and use that if it is
    if spatial_metric_filepath(simulation_directory, SpatialMetric.DBT_EVAP).exists():
        dbt = dbt_evap(simulation_directory, epw)
    else:
        dbt = dbt_epw(simulation_directory, epw)

    if spatial_metric_filepath(simulation_directory, SpatialMetric.RH_EVAP).exists():
        rh = rh_evap(simulation_directory, epw)
    else:
        rh = rh_epw(simulation_directory, epw)

    # check that CFD wind speeds are available, and use that if it is
    if spatial_metric_filepath(simulation_directory, SpatialMetric.WS_CFD).exists():
        ws = ws_cfd(simulation_directory, epw)
    else:
        ws = ws_epw(simulation_directory, epw)

    mrt = mrt_interpolated(
        simulation_directory,
        unshaded_mean_radiant_temperature,
        shaded_mean_radiant_temperature,
        total_irradiance,
        sky_view,
        epw,
    )

    utci_df = utci(dbt, rh, mrt, ws)
    utci_df.to_hdf(utci_path, "df", complevel=9, complib="blosc")

    return utci_df
