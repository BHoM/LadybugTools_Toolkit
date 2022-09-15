from pathlib import Path
from typing import Union

import pandas as pd
from ladybugtools_toolkit.honeybee_extension.results.load_sun_up_hours import (
    load_sun_up_hours,
)


from python_toolkit.bhom.analytics import analytics


@analytics
def load_ill_file(ill_file: Union[str, Path]) -> pd.Series:
    """Load a Radiance .ill file and return a DataFrame with the data.

    Args:
        ill_file (Union[str, Path]): The path to the Radiance .ill file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .ill file.
    """
    ill_file = Path(ill_file)
    sun_up_hours_file = ill_file.parent / "sun-up-hours.txt"
    df = pd.read_csv(ill_file, sep=r"\s+", header=None, index_col=None).T
    df.columns = pd.MultiIndex.from_product([[ill_file.stem], df.columns])
    df.index = load_sun_up_hours(sun_up_hours_file)
    return df
