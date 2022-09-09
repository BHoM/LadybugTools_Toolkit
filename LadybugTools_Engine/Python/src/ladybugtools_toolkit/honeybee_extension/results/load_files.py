import warnings
from pathlib import Path
from typing import Callable, List, Union

import pandas as pd


def load_files(func: Callable, files: List[Union[str, Path]]) -> pd.DataFrame:
    """Load a set of input files and combine into a DataFrame with filename as header.

    Args:
        input_files (List[Union[str, Path]]): A list of paths to the input files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input files.
    """
    if isinstance(files, (str, Path)):
        files = [files]

    if len(files) == 0:
        raise FileNotFoundError("No files of the specified type were found.")

    filenames = [Path(i).stem for i in files]
    if len(set(filenames)) != len(filenames):
        err_str = f"There are duplicate filenames in the list of input files for {func.__name__}. This may cause issues when trying to reference specific results sets!"
        warnings.warn(err_str)

    return pd.concat([func(i) for i in files], axis=1).sort_index(axis=1)
