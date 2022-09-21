import datetime
from pathlib import Path
from typing import Union

import pandas as pd


from ladybugtools_toolkit import analytics


@analytics
def load_sun_up_hours(
    sun_up_hours_file: Union[str, Path], year: int = 2017
) -> pd.DatetimeIndex:
    """Load a HB-Radiance generated sun-up-hours.txt file and return a DatetimeIndex with the data.

    Args:
        sun_up_hours_file (Union[str, Path]): Path to the sun-up-hours.txt file.
        year (int, optional): The year to be used to generate the DatetimeIndex. Defaults to 2017.

    Returns:
        pd.DatetimeIndex: A Pandas DatetimeIndex with the data from the sun-up-hours.txt file.
    """

    sun_up_hours_file = Path(sun_up_hours_file)
    float_hours = pd.read_csv(sun_up_hours_file, header=None, index_col=None).squeeze()
    start_date = datetime.datetime(year, 1, 1, 0, 0, 0)
    timedeltas = [datetime.timedelta(hours=i) for i in float_hours]
    index = pd.DatetimeIndex([start_date + i for i in timedeltas])

    return index
