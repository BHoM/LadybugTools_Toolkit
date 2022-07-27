from pathlib import Path
from typing import List, Union

import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.calculate_evaporatively_cooled_dbt_rh import (
    calculate_evaporatively_cooled_dbt_rh,
)
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.moisture_directory import (
    moisture_directory as md,
)


def load_relative_humidity_moisture(
    simulation_directory: Union[str, Path],
    spatial_points: List[List[float]],
    epw: EPW,
) -> pd.DataFrame:
    """Read (or calculate) evaporatively cooled RH.

    Args:
        simulation_directory (Union[str, Path]):
            The directory in which spatial simulation results exist.
        spatial_points (List[List[float]]):
            A list of [[x0, y0], [x1, y1], ...] points locations.
        epw (EPW):
            A Ladybug EPW object.

    Returns:
        pd.DataFrame:
            A DataFrame containing spatial RH adjusted by moisture addition into the air.
    """

    moisture_directory = md(simulation_directory)
    rh_path = moisture_directory / "relative_humidity_evap.h5"

    if rh_path.exists():
        print(
            f"- Loading RH (evaporatively cooled) data from {simulation_directory.name}"
        )
        return pd.read_hdf(rh_path, "df")

    print(f"- Processing evaporatively cooled RH data for {simulation_directory.name}")

    _, rh = calculate_evaporatively_cooled_dbt_rh(
        simulation_directory, spatial_points, epw
    )

    return rh
