from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.unshaded_shaded_interpolation import (
    unshaded_shaded_interpolation,
)


def load_universal_thermal_climate_index_interpolated(
    simulation_directory: Union[str, Path],
    unshaded_universal_thermal_climate_index: HourlyContinuousCollection,
    shaded_universal_thermal_climate_index: HourlyContinuousCollection,
    total_irradiance: pd.DataFrame,
    sky_view: pd.DataFrame,
    epw: EPW,
) -> pd.DataFrame:
    """Return the UTCI from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Union[str, Path]): The simulation directory.
            containing unshaded UTCI values.
        shaded_universal_thermal_climate_index (HourlyContinuousCollection): A collection containing
            shaded UTCI values.
        total_irradiance (pd.DataFrame): A dataframe containing spatial total irradiance values.
        sky_view (pd.DataFrame): A dataframe containing point-wise sky-view values.
        epw (EPW): The associate EPW file.

    Returns:
        pd.DataFrame: A dataframe containing spatial MRT values.
    """

    simulation_directory = Path(simulation_directory)

    utci_path = simulation_directory / "universal_thermal_climate_index_interpolated.h5"

    if utci_path.exists():
        print(f"- Loading UTCI (interpolated) data from {simulation_directory.name}")
        return pd.read_hdf(utci_path, "df")

    print(f"- Processing UTCI (interpolated) data for {simulation_directory.name}")

    utci = unshaded_shaded_interpolation(
        unshaded_universal_thermal_climate_index,
        shaded_universal_thermal_climate_index,
        total_irradiance,
        sky_view,
        np.array(epw.global_horizontal_radiation.values) > 0,
    )

    utci.to_hdf(utci_path, "df", complevel=9, complib="blosc")
    return utci
