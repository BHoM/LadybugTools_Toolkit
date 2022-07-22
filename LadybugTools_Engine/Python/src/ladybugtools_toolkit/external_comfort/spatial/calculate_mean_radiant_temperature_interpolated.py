from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.unshaded_shaded_interpolation import (
    unshaded_shaded_interpolation,
)


def calculate_mean_radiant_temperature_interpolated(
    simulation_directory: Union[str, Path],
    unshaded_mean_radiant_temperature: HourlyContinuousCollection,
    shaded_mean_radiant_temperature: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view: pd.DataFrame,
    epw: EPW,
) -> pd.DataFrame:
    """Return the MRT from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Union[str, Path]): The simulation directory.
        unshaded_mean_radiant_temperature (HourlyContinuousCollection): A collection containing
            unshaded MRT values.
        shaded_mean_radiant_temperature (HourlyContinuousCollection): A collection containing
            shaded MRT values.
        total_irradiance (pd.DataFrame): A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame): A dataframe containing point-wise sky-view values.
        epw (EPW): The associate EPW file.

    Returns:
        pd.DataFrame: A dataframe containing spatial MRT values.
    """

    simulation_directory = Path(simulation_directory)

    mrt_path = simulation_directory / "mean_radiant_temperature_interpolated.h5"

    if mrt_path.exists():
        print(
            f"- Loading mean-radiant-temperature data from {simulation_directory.name}"
        )
        return pd.read_hdf(mrt_path, "df")

    print(f"- Processing mean-radiant-temperature data for {simulation_directory.name}")

    mrt = unshaded_shaded_interpolation(
        unshaded_mean_radiant_temperature,
        shaded_mean_radiant_temperature,
        total_irradiance,
        sky_view,
        np.array(epw.global_horizontal_radiation.values) > 0,
    )

    mrt.to_hdf(mrt_path, "df", complevel=9, complib="blosc")

    return mrt
