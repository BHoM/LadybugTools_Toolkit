from pathlib import Path
from typing import Union

import pandas as pd
from ladybugtools_toolkit.honeybee_extension.results.load_pts import load_pts


def load_points(simulation_directory: Union[str, Path]) -> pd.DataFrame:
    """Return the points results from the simulation directory, and create the H5 file to store
        them as compressed objects if not already done.

    Args:
        simulation_directory (Union[str, Path]): The directory containing a pts file.

    Returns:
        pd.DataFrame: A dataframe with the points locations.
    """

    simulation_directory = Path(simulation_directory)

    points_path = simulation_directory / "points.h5"

    if points_path.exists():
        print(f"- Loading points data from {simulation_directory.name}")
        return pd.read_hdf(points_path, "df")

    print(f"- Processing points data for {simulation_directory.name}")
    points_files = list(
        (simulation_directory / "sky_view" / "model" / "grid").glob("*.pts")
    )
    points = load_pts(points_files)
    points.to_hdf(points_path, "df", complevel=9, complib="blosc")
    return points
