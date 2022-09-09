from pathlib import Path

import pandas as pd


def load_cfd_extract(file: Path, value_renamer: str = "VELOCITY") -> pd.DataFrame:
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
    df = (
        pd.read_csv(file, skiprows=5, index_col=0)
        .dropna(how="any")
        .rename(columns={" X [ m ]": "x", " Y [ m ]": "y", " Z [ m ]": "z"})
    )
    # rename last column
    df.columns = [*df.columns[:-1], value_renamer]

    if len(df.columns) > 4:
        raise ValueError(
            "The number of variables inside the given file exceeds the number possible "
            + "(there should only be Node Number, X [ m ], Y [ m ], Z [ m ], <VEL> [ m s^-1 ])."
        )

    return df
