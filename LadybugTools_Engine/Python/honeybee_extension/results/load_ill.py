import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
from typing import List, Union

import pandas as pd
from honeybee_extension.results.load_files import load_files
from honeybee_extension.results.load_sun_up_hours import load_sun_up_hours


def _load_ill_file(ill_file: Union[str, Path]) -> pd.Series:
    """Load a Radiance .ill file and return a DataFrame with the data.

    Args:
        ill_file (Union[str, Path]): The path to the Radiance .ill file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .ill file.
    """
    ill_file = Path(ill_file)
    sun_up_hours_file = ill_file.parent / "sun-up-hours.txt"
    df = pd.read_csv(ill_file, sep="\s+", header=None, index_col=None).T
    df.columns = pd.MultiIndex.from_product([[ill_file.stem], df.columns])
    df.index = load_sun_up_hours(sun_up_hours_file)
    return df


def load_ill(ill_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .ill file, or list of Radiance .ill files and return a combined DataFrame with the data.

    Args:
        ill_files (Union[str, Path, List[Union[str, Path]]]): A single .ill file, or a list of .ill files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .ill files.
    """
    return load_files(_load_ill_file, ill_files)
