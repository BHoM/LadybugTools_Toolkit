from pathlib import Path
from typing import List, Union

import pandas as pd

from .load_files import load_files
from .load_ill_file import _load_ill_file


def load_ill(ill_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .ill file, or list of Radiance .ill files and return a combined DataFrame with the data.

    Args:
        ill_files (Union[str, Path, List[Union[str, Path]]]): A single .ill file, or a list of .ill files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .ill files.
    """
    return load_files(_load_ill_file, ill_files)
