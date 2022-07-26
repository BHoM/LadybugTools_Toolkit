from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.load_points import load_points
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def create_epw_dbt_matrix(
    simulation_directory: Union[str, Path],
    epw: EPW = None,
) -> pd.DataFrame:
    """Return the dry-bulb-temperatures from the simulation directory, and create the H5 file to
        store them as compressed objects if not already done.

    Args:
        simulation_directory (Union[str, Path]): The directory containing simulation results.
        epw (EPW): The associated EPW file.

    Returns:
        pd.DataFrame: A dataframe with the spatial dry-bulb-temperature.
    """

    simulation_directory = Path(simulation_directory)

    spatial_points = load_points(simulation_directory)
    n_pts = len(spatial_points.columns)

    dbt_path = simulation_directory / "dry_bulb_temperature.h5"

    if dbt_path.exists():
        print(f"- Loading DBT data from {simulation_directory.name}")
        return pd.read_hdf(dbt_path, "df")

    print(f"- Processing dry-bulb temperature data for {simulation_directory.name}")

    dbt_series = to_series(epw.dry_bulb_temperature)
    dbt_df = pd.DataFrame(
        np.tile(dbt_series.values, (n_pts, 1)).T, index=dbt_series.index
    )

    dbt_df.to_hdf(dbt_path, "df", complevel=9, complib="blosc")

    return dbt_df
