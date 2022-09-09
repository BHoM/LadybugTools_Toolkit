from pathlib import Path
from typing import List, Union

import pandas as pd

from ladybugtools_toolkit.honeybee_extension.results.load_files import load_files
from ladybugtools_toolkit.honeybee_extension.results.load_res_file import load_res_file


def load_res(res_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .res file, or list of Radiance .res files and return a combined DataFrame with the data.

    NOTE: This also works with daylight metrics files (da, cda, udi, udi_lower and udi_upper).

    Args:
        res_files (Union[str, Path, List[Union[str, Path]]]): A single .res file, or a list of .res files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .res files.
    """
    return load_files(load_res_file, res_files)
