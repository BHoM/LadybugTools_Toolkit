import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from pathlib import Path
from typing import List, Union

import pandas as pd
from honeybee_extension.results.load_files import load_files


def _load_res_file(res_file: Union[str, Path]) -> pd.Series:
    """Load a Radiance .res file and return a DataFrame with the data. 
    
    NOTE: This also works with daylight metrics files (da, cda, udi, udi_lower and udi_upper).

    Args:
        res_file (Union[str, Path]): The path to the Radiance .res file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .res file.
    """
    res_file = Path(res_file)
    series = pd.read_csv(res_file, header=None, sep="\s+").squeeze()
    series.name = res_file.stem
    return series


def load_res(res_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .res file, or list of Radiance .res files and return a combined DataFrame with the data. 
    
    NOTE: This also works with daylight metrics files (da, cda, udi, udi_lower and udi_upper).

    Args:
        res_files (Union[str, Path, List[Union[str, Path]]]): A single .res file, or a list of .res files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .res files.
    """
    return load_files(_load_res_file, res_files)
