from pathlib import Path
from typing import List, Union

import pandas as pd

from .load_files import load_files
from .load_sql_file import load_sql_file


def load_sql(sql_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single EnergyPlus .sql file, or list of EnergyPlus .sql files and return a combined DataFrame with the data.

    Args:
        sql_files (Union[str, Path, List[Union[str, Path]]]): A single .sql file, or a list of .sql files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .sql files.
    """
    return load_files(load_sql_file, sql_files)
