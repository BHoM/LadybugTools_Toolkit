import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
from typing import List, Union

import pandas as pd
from honeybee_extension.results.load_files import load_files


def _load_pts_file(pts_file: Union[str, Path]) -> pd.DataFrame:
    """Load a Radiance .pts file and return a DataFrame with the data.

    Args:
        pts_file (Union[str, Path]): The path to the Radiance .pts file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .pts file.
    """
    pts_file = Path(pts_file)
    df = pd.read_csv(
        pts_file, header=None, names=["x", "y", "z", "vx", "vy", "vz"], sep="\s+"
    )
    df.columns = pd.MultiIndex.from_product([[pts_file.stem], df.columns])
    return df


def load_pts(pts_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .pts file, or list of Radiance .pts files and return a combined DataFrame with the data.

    Args:
        pts_files (Union[str, Path, List[Union[str, Path]]]): A single .pts file, or a list of .pts files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .pts files.
    """
    return load_files(_load_pts_file, pts_files)
