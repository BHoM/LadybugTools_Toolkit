from pathlib import Path
from typing import Union

import pandas as pd


def load_res_file(res_file: Union[str, Path]) -> pd.Series:
    """Load a Radiance .res file and return a DataFrame with the data.

    NOTE: This also works with daylight metrics files (da, cda, udi, udi_lower and udi_upper).

    Args:
        res_file (Union[str, Path]): The path to the Radiance .res file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .res file.
    """
    res_file = Path(res_file)
    series = pd.read_csv(res_file, header=None, sep=r"\s+").squeeze()
    series.name = res_file.stem
    return series
