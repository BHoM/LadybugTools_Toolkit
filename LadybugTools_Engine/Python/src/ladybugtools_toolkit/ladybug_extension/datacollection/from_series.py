import pandas as pd
from ladybug._datacollectionbase import BaseCollection
from ladybug.datacollection import HourlyContinuousCollection, MonthlyCollection
from ladybugtools_toolkit.ladybug_extension.header.from_string import from_string


from python_toolkit.bhom.analytics import analytics


@analytics
def from_series(series: pd.Series) -> BaseCollection:
    """Convert a Pandas Series object into a Ladybug BaseCollection-like object.

    Args:
        series (pd.Series): A Pandas Series object.

    Returns:
        BaseCollection: A Ladybug BaseCollection-like object.
    """

    leap_yr = True if series.index.is_leap_year.any() else False

    header = from_string(series.name)
    header.metadata["source"] = "From custom pd.Series"

    freq = pd.infer_freq(series.index)
    if freq in ["H"]:
        if leap_yr:
            if len(series.index) != 8784:
                raise ValueError(
                    "The number of values in the series must be 8784 for leap years."
                )
        else:
            if len(series.index) != 8760:
                raise ValueError("The series must have 8760 rows for non-leap years.")

        return HourlyContinuousCollection(
            header=header,
            values=series.values,
        )
    elif freq in ["M", "MS"]:
        if len(series.index) != 12:
            raise ValueError("The series must have 12 rows for months.")

        return MonthlyCollection(
            header=header,
            values=series.values,
            datetimes=range(1, 13),
        )
    else:
        raise ValueError("The series must be hourly or monthly.")
