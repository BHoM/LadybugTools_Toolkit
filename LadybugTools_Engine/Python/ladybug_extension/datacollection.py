import numpy as np
import pandas as pd
from ladybug._datacollectionbase import BaseCollection
from ladybug.datacollection import HourlyContinuousCollection, MonthlyCollection

from .analysisperiod import to_datetimes
from .header import from_string, to_string


def to_series(collection: BaseCollection) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object."""

    index = to_datetimes(collection.header.analysis_period)
    if len(collection.values) == 12:
        index = pd.date_range(f"{index[0].year}-01-01", periods=12, freq="MS")

    return pd.Series(
        data=collection.values,
        index=index,
        name=to_string(collection.header),
    )


def from_series(series: pd.Series) -> BaseCollection:
    """Convert a Pandas Series object into a Ladybug BaseCollection-like object."""

    if series.index.is_leap_year.any():
        leap_yr = True
    else:
        leap_yr = False

    header = from_string(series.name)
    header.metadata["source"] = "From custom pd.Series"

    freq = pd.infer_freq(series.index)
    if freq in ["H"]:
        if leap_yr:
            assert (
                len(series.index) == 8784
            ), "The number of values in the series must be 8784 for leap years."
        else:
            assert (
                len(series.index) == 8760
            ), "The series must have 8760 rows for non-leap years."

        return HourlyContinuousCollection(
            header=header,
            values=series.values,
        )
    elif freq in ["M", "MS"]:
        assert len(series.index) == 12, "The series must have 12 rows for months."

        return MonthlyCollection(
            header=header,
            values=series.values,
            datetimes=range(1, 13),
        )
    else:
        raise ValueError("The series must be hourly or monthly.")


def to_array(collection: BaseCollection) -> np.ndarray:
    """Convert a Ladybug BaseCollection-like object into a numpy array."""

    return np.array(collection.values)


def to_hourly(
    collection: MonthlyCollection, method: str = None
) -> HourlyContinuousCollection:
    """Resample a Ladybug MonthlyContinuousCollection object into a Ladybug HourlyContinuousCollection object.

    Args:
        method (str): The method to use for annualizing the monthly values.

    Returns:
        HourlyContinuousCollection: A Ladybug HourlyContinuousCollection object.
    """

    if method is None:
        method = "smooth"

    interpolation_methods = {
        "smooth": "quadratic",
        "step": "pad",
        "linear": "linear",
    }

    series = to_series(collection)
    annual_hourly_index = pd.date_range(
        f"{series.index[0].year}-01-01", periods=8760, freq="H"
    )
    series_annual = series.reindex(annual_hourly_index)
    series_annual[series_annual.index[-1]] = series_annual[series_annual.index[0]]

    try:
        return HourlyContinuousCollection(
            header=collection.header,
            values=series_annual.interpolate(
                method=interpolation_methods[method]
            ).values,
        )
    except KeyError as e:
        raise e
