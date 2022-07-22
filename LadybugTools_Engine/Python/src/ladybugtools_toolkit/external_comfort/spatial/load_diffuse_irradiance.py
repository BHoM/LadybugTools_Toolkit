from pathlib import Path
from typing import Union

import pandas as pd
from ladybugtools_toolkit.external_comfort.spatial.load_direct_irradiance import \
    load_direct_irradiance
from ladybugtools_toolkit.external_comfort.spatial.load_total_irradiance import \
    load_total_irradiance


def load_diffuse_irradiance(
    simulation_directory: Union[str, Path],
    total_irradiance: pd.DataFrame = None,
    direct_irradiance: pd.DataFrame = None,
) -> pd.DataFrame:
    """Return the diffuse irradiance from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Union[str, Path]): The directory containing a diffuse irradiance ILL
            file.
        total_irradiance (pd.DataFrame, optional): If given along with
            direct_irradiance, then calculation will be completed faster. Default is None.
        direct_irradiance (pd.DataFrame, optional): If given along with
            total_irradiance, then calculation will be completed faster. Default is None.

    Returns:
        pd.DataFrame: A dataframe with the diffuse irradiance.
    """

    simulation_directory = Path(simulation_directory)

    diffuse_irradiance_path = simulation_directory / "diffuse_irradiance.h5"

    if diffuse_irradiance_path.exists():
        print(f"- Loading diffuse irradiance data from {simulation_directory.name}")
        return pd.read_hdf(diffuse_irradiance_path, "df")

    print(f"- Processing diffuse irradiance data for {simulation_directory.name}")
    if (total_irradiance is None) and (direct_irradiance is None):
        total_irradiance = load_total_irradiance(simulation_directory)
        direct_irradiance = load_direct_irradiance(simulation_directory)

    diffuse_irradiance = total_irradiance - direct_irradiance

    diffuse_irradiance.clip(lower=0).to_hdf(
        diffuse_irradiance_path, "df", complevel=9, complib="blosc"
    )
    return diffuse_irradiance
