from pathlib import Path
from typing import List, Union

import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.calculate_evaporatively_cooled_dbt_rh import \
    calculate_evaporatively_cooled_dbt_rh
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.moisture_directory import \
    moisture_directory as md


def load_dry_bulb_temperature_moisture(
    simulation_directory: Union[str, Path],
    spatial_points: List[List[float]],
    epw: EPW = None,
) -> pd.DataFrame:

    moisture_directory = md(simulation_directory)
    dbt_path = moisture_directory / "dry_bulb_temperature_evap.h5"

    if dbt_path.exists():
        print(
            f"- Loading DBT (evaporatively cooled) data from {simulation_directory.name}"
        )
        return pd.read_hdf(dbt_path, "df")

    print(f"- Processing evaporatively cooled DBT data for {simulation_directory.name}")

    dbt, _ = calculate_evaporatively_cooled_dbt_rh(
        simulation_directory, spatial_points, epw
    )

    return dbt
