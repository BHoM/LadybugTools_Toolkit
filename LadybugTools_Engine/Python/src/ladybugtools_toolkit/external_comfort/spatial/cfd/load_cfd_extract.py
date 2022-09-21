from io import StringIO
from pathlib import Path

import pandas as pd
from ladybugtools_toolkit import analytics


@analytics
def load_cfd_extract(file: Path, velocity_col: str = "VELOCITY") -> pd.DataFrame:
    """Load a file containing an extract from CFX-Post, with point XYZ values,
        and variable for that point.

    Args:
        file (Path):
            The path to the CFX-Post extract file.
        value_renamer (str, optional):
            A string to rename the last column by.

    Returns:
        pd.DataFrame:
            A DataFrame containing the data from the given file.
    """
    
    with open(file, "r", encoding="utf-8") as fp:
        dat = fp.read()
    
    df = pd.read_csv(StringIO(dat.split("\n\n")[1]), sep=",", header=[0, 1]).reset_index().drop(columns=["level_0"])
    if len(df.columns) != 4:
        raise ValueError("Columns should be of length 4 (x, y, z, velocity)")
    
    df.columns = ["x", "y", "z", velocity_col]

    # replace any null values with the avg for the whole dataset
    df[velocity_col] = pd.to_numeric(df[velocity_col], errors="coerce")
    df[velocity_col] = df[velocity_col].fillna(df[velocity_col].mean())

    return df
