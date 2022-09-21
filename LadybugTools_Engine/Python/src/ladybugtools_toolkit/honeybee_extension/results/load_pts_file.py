from pathlib import Path
from typing import Union

import pandas as pd


from ladybugtools_toolkit import analytics


@analytics
def load_pts_file(pts_file: Union[str, Path]) -> pd.DataFrame:
    """Load a Radiance .pts file and return a DataFrame with the data.

    Args:
        pts_file (Union[str, Path]): The path to the Radiance .pts file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .pts file.
    """
    pts_file = Path(pts_file)
    df = pd.read_csv(
        pts_file, header=None, names=["x", "y", "z", "vx", "vy", "vz"], sep=r"\s+"
    )
    df.columns = pd.MultiIndex.from_product([[pts_file.stem], df.columns])
    return df
