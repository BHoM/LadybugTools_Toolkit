from pathlib import Path
from typing import Union

import pandas as pd
from ladybugtools_toolkit.honeybee_extension.results.load_res import load_res


def load_sky_view(simulation_directory: Union[str, Path]) -> pd.DataFrame:
    """Get the sky view from the simulation directory.

    Args:
        simulation_directory (Union[str, Path]): The simulation directory containing a sky view RES
            file.

    Returns:
        pd.DataFrame: The sky view dataframe.
    """

    simulation_directory = Path(simulation_directory)

    sky_view_path = simulation_directory / "sky_view.h5"

    if sky_view_path.exists():
        print(f"- Loading sky-view data from {simulation_directory.name}")
        return pd.read_hdf(sky_view_path, "df")

    print(f"- Processing sky-view data for {simulation_directory.name}")
    res_files = list((simulation_directory / "sky_view" / "results").glob("*.res"))
    sky_view = load_res(res_files).clip(lower=0, upper=100)
    sky_view.to_hdf(sky_view_path, "df", complevel=9, complib="blosc")
    return sky_view
