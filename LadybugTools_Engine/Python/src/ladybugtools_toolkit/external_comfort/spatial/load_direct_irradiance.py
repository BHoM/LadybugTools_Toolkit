from pathlib import Path
from typing import Union

import pandas as pd
from ladybugtools_toolkit.honeybee_extension.results.load_ill import load_ill
from ladybugtools_toolkit.honeybee_extension.results.make_annual import make_annual


def load_direct_irradiance(simulation_directory: Union[str, Path]) -> pd.DataFrame:
    """Return the direct irradiance from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Union[str, Path]): The directory containing a direct irradiance ILL
            file.

    Returns:
        pd.DataFrame: A dataframe with the direct irradiance.
    """

    simulation_directory = Path(simulation_directory)

    direct_irradiance_path = simulation_directory / "direct_irradiance.h5"

    if direct_irradiance_path.exists():
        print(f"- Loading direct irradiance data from {simulation_directory.name}")
        return pd.read_hdf(direct_irradiance_path, "df")

    print(f"- Processing direct irradiance data for {simulation_directory.name}")
    ill_files = list(
        (simulation_directory / "annual_irradiance" / "results" / "direct").glob(
            "*.ill"
        )
    )
    direct_irradiance = make_annual(load_ill(ill_files)).fillna(0)

    direct_irradiance.clip(lower=0).to_hdf(
        direct_irradiance_path, "df", complevel=9, complib="blosc"
    )
    return direct_irradiance
