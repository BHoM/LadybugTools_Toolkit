"""Methods for manipulating Ladybug data collections."""

# pylint: disable=E0401
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# pylint: enable=E0401

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    BaseCollection,
    HourlyContinuousCollection,
    MonthlyCollection,
)
from ladybug.datatype.angle import Angle
from ladybug.dt import DateTime
from ..bhom.analytics import bhom_analytics
from ..helpers import circular_weighted_mean
from .analysisperiod import analysis_period_to_datetimes
from .analysisperiod import describe_analysis_period
from .header import header_from_string, header_to_string


def collection_to_series(collection: BaseCollection, name: str = None) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.

    Args:
        collection (BaseCollection):
            Ladybug data collection object.
        name (str, optional):
            The name of the resulting Pandas Series object. Defaults to None,
            which uses the collection datatype.

    Returns:
        pd.Series:
            A Pandas Series object.
    """

    index = analysis_period_to_datetimes(collection.header.analysis_period)
    if len(collection.values) == 12:
        index = pd.date_range(f"{index[0].year}-01-01", periods=12, freq="MS")

    return pd.Series(
        data=collection.values,
        index=index,
        name=header_to_string(collection.header) if not name else name,
    )


def collection_from_series(series: pd.Series) -> BaseCollection:
    """Convert a Pandas Series object into a Ladybug BaseCollection-like object.

    Args:
        series (pd.Series): A Pandas Series object.

    Returns:
        BaseCollection: A Ladybug BaseCollection-like object.
    """

    header = header_from_string(
        series.name, is_leap_year=series.index.is_leap_year.any()
    )
    header.metadata["source"] = "From custom pd.Series"

    freq = pd.infer_freq(series.index)
    if freq in ["H", "h"]:
        if series.index.is_leap_year.any():
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

    if freq in ["M", "MS"]:
        if len(series.index) != 12:
            raise ValueError("The series must have 12 rows for months.")

        return MonthlyCollection(
            header=header,
            values=series.values.tolist(),
            datetimes=range(1, 13),
        )

    raise ValueError("The series must be hourly or monthly.")


@bhom_analytics()
def percentile(
    collections: list[BaseCollection], nth_percentile: float
) -> BaseCollection:
    """Create an nth percentile of the given data collections.

    Args:
        collections (list[BaseCollection]):
            A list of collections.
        nth_percentile (float):
            A percentile, between 0 and 1.

    Returns:
        BaseCollection:
            The "nth percentile" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([collection_to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError(
            'You cannot get the "nth percentile" across non-alike datatypes.'
        )

    # check if any collections input are angular, in which case, fail
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        raise ValueError("This method cannot be applied to Angular datatypes.")

    return collection_from_series(
        df.quantile(nth_percentile, axis=1).rename(series_name)
    )


@bhom_analytics()
def minimum(collections: list[BaseCollection]) -> BaseCollection:
    """Create a Minimum of the given data collections.

    Args:
        collections (list[BaseCollection]):
            A list of collections.

    Returns:
        BaseCollection:
            The "Minimum" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([collection_to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError('You cannot get the "minimum" across non-alike datatypes.')

    # check if any collections input are angular, in which case, fail
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        raise ValueError("This method cannot be applied to Angular datatypes.")
    return collection_from_series(df.min(axis=1).rename(series_name))


@bhom_analytics()
def maximum(collections: list[BaseCollection]) -> BaseCollection:
    """Create a Maximum of the given data collections.

    Args:
        collections (list[BaseCollection]):
            A list of collections.

    Returns:
        BaseCollection:
            The "Maximum" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([collection_to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError('You cannot get the "maximum" across non-alike datatypes.')

    # check if any collections input are angular, in which case, fail
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        raise ValueError("This method cannot be applied to Angular datatypes.")
    return collection_from_series(df.max(axis=1).rename(series_name))


@bhom_analytics()
def summarise_collection(
    collection: BaseCollection,
    _n_common: int = 3,
) -> list[str]:
    """Describe a datacollection.

    Args:
        ...

    Returns:
        list[str]: A list of strings describing this data collection.

    """
    # pylint: disable=C0301
    descriptions = []

    series = collection_to_series(collection)

    name = " ".join(series.name.split(" ")[:-1])
    unit = series.name.split(" ")[-1].replace("(", "").replace(")", "")
    period_ts = describe_analysis_period(
        collection.header.analysis_period, include_timestep=True
    )
    period = describe_analysis_period(
        collection.header.analysis_period, include_timestep=False
    )

    if (collection.header.analysis_period.st_month != 1) or (
        collection.header.analysis_period.end_month != 12
    ):
        warnings.warn(
            "The data collection being described represents only a portion of a full year."
        )

    # summarise data title and time period
    descriptions.append(
        f"{name}, for {period_ts} (representing {len(series)} values, each in units of {unit})."
    )

    # determine if data is time-unit modifiable
    rate_of_change = False
    try:
        collection.to_time_rate_of_change()
        rate_of_change = True
    except (AssertionError, ValueError):
        pass

    if not isinstance(collection.header.data_type, Angle):
        # minimum value (number of times it occurs, and when)
        _min = series.min()
        _n_min = len(series[series == _min])
        _min_mean_time = series.groupby(series.index.time).mean().idxmin()
        descriptions.append(
            f"Minimum {name} is {_min:,.01f}{unit}, occurring {'once' if _n_min == 1 else f'{_n_min} times'}{f' and typically at {_min_mean_time:%H:%M}' if _n_min > 1 else f' on {series.idxmin():%b %d at %H:%M}'}."
        )

        # 25%ile
        _lower = series.quantile(0.25)
        _n_less = len(series[series < _lower])
        descriptions.append(
            f"25%-ile {name} is {_lower:,.01f}{unit}, with {_n_less} occurrences below that value."
        )

    if isinstance(collection.header.data_type, Angle):
        _mean = circular_weighted_mean(series.values)
        descriptions.append(f"Mean {name} is {_mean:,.01f}{unit}.")
    else:
        # mean value
        _mean = series.mean()
        _std = series.std()
        descriptions.append(
            f"Mean {name} is {_mean:,.01f}{unit}, with a standard deviation of {_std:0.1f}{unit}."
        )

    if not isinstance(collection.header.data_type, Angle):
        # median value
        _median = series.median()
        _n_median = len(series[series == _median])
        descriptions.append(f"Median {name} is {_median:,.01f}{unit}.")

        # 75%ile
        _upper = series.quantile(0.75)
        _n_more = len(series[series > _upper])
        descriptions.append(
            f"75%-ile {name} is {_upper:,.01f}{unit}, with {_n_more} occurrences above that value."
        )

        # maximum value (number of times it occurs, and when)
        _max = series.max()
        _n_max = len(series[series == _max])
        _max_mean_time = series.groupby(series.index.time).mean().idxmax()
        descriptions.append(
            f"Maximum {name} is {_max:,.01f}{unit}, "
            f"occurring {_n_max} time"
            f"{f's and typically at {_max_mean_time:%H:%M}' if _n_max > 1 else f' on {series.idxmax():%b %d at %H:%M}'}."
        )

    # common values
    _most_common, _most_common_counts = series.value_counts().reset_index().values.T
    _most_common_str = (
        " and ".join(
            f"{unit}, ".join([f"{i:,.0f}" for i in _most_common[:_n_common]]).rsplit(
                ", ", 1
            )
        )
        + unit
    )
    _most_common_counts_str = (
        " and ".join(
            ", ".join(_most_common_counts[:_n_common].astype(int).astype(str)).rsplit(
                ", ", 1
            )
        )
        + " times respectively"
    )
    descriptions.append(
        f"The {_n_common} most common {name} values are {_most_common_str}, occurring {_most_common_counts_str}."
    )

    if not isinstance(collection.header.data_type, Angle):
        # cumulative values
        if rate_of_change:
            descriptions.append(
                f"Cumulative total {name} for {period} is "
                f"{series.sum():,.0f}{collection.to_time_rate_of_change().header.unit}."
            )

        # agg months
        _month_mean = series.resample("MS").mean()
        descriptions.append(
            f"The month with the lowest mean {name} is "
            f"{_month_mean.idxmin():%B}, with a value of {_month_mean.min():,.1f}{unit}."
        )
        descriptions.append(
            f"The month with the highest mean {name} is "
            f"{_month_mean.idxmax():%B}, with a value of {_month_mean.max():,.1f}{unit}."
        )

        # agg times
        _time_mean = series.groupby(series.index.time).mean()
        descriptions.append(
            f"The time when the highest mean {name} typically occurs "
            f"is {_time_mean.idxmax():%H:%M}, with a value of {_time_mean.max():,.1f}{unit}."
        )
        descriptions.append(
            f"The time when the lowest mean {name} typically occurs "
            f"is {_time_mean.idxmin():%H:%M}, with a mean value of {_time_mean.min():,.1f}{unit}."
        )

        # diurnal
        _month_grp = series.resample("MS")
        _month_grp_min = _month_grp.min()
        _month_grp_max = _month_grp.max()
        _month_range_month_avg = _month_grp_max - _month_grp_min
        descriptions.append(
            f"The month when the largest range of {name} typically occurs "
            f"is {_month_range_month_avg.idxmax():%B}, with values between "
            f"{_month_grp_min.loc[_month_range_month_avg.idxmax()]:,.1f}{unit} "
            f"and {_month_grp_max.loc[_month_range_month_avg.idxmax()]:,.1f}{unit}."
        )
        descriptions.append(
            f"The month when the smallest range of {name} typically occurs "
            f"is {_month_range_month_avg.idxmin():%B}, with values between "
            f"{_month_grp_min.loc[_month_range_month_avg.idxmin()]:,.1f}{unit} "
            f"and {_month_grp_max.loc[_month_range_month_avg.idxmin()]:,.1f}{unit}."
        )
    # pylint: enable=C0301
    return descriptions


@bhom_analytics()
def average(
    collections: list[BaseCollection], weights: list[float] = None
) -> BaseCollection:
    """Create an Average of the given data collections.

    Args:
        collections (list[BaseCollection]):
            A list of collections.
        weights (list[float], optional):
            A list of weights for each collection.
            Defaults to None which evenly weights each collection.

    Returns:
        BaseCollection:
            The "Average" collection.
    """

    if len(collections) == 1:
        return collections[0]

    for col in collections[1:]:
        if col.header.data_type != collections[0].header.data_type:
            raise ValueError(
                "You cannot get the average across non-alike datatypes "
                f"({col.header.unit} != {collections[0].header.unit})."
            )

    if weights is None:
        weights = np.ones(len(collections)) / len(collections)
    weights = np.array(weights)
    if len(weights) != len(collections):
        raise ValueError("The number of weights must match the number of collections.")
    if (weights < 0).sum():
        raise ValueError("Weights must be positive.")

    # construct df
    df = pd.concat([collection_to_series(i) for i in collections], axis=1)
    vals = None
    if isinstance(collections[0].header.data_type, Angle):
        vals = []
        for _, row in df.iterrows():
            vals.append(circular_weighted_mean(row, weights))
    else:
        vals = (
            df.groupby(df.index)
            .apply(lambda x: np.average(x, weights=weights, axis=1)[0])
            .values.tolist()
        )

    return collections[0].get_aligned_collection(vals)


@bhom_analytics()
def to_hourly(
    collection: MonthlyCollection, method: str = None
) -> HourlyContinuousCollection:
    """
    Resample a Ladybug MonthlyContinuousCollection object into a Ladybug
    HourlyContinuousCollection object.

    Args:
        collection (MonthlyCollection):
            A Ladybug MonthlyContinuousCollection object.
        method (str):
            The method to use for annualizing the monthly values.

    Returns:
        HourlyContinuousCollection:
            A Ladybug HourlyContinuousCollection object.
    """
    if method is None:
        method = "smooth"

    interpolation_methods = {
        "smooth": "quadratic",
        "step": "pad",
        "linear": "linear",
    }

    if method not in interpolation_methods:
        raise ValueError(
            f"The up-sampling method must be one of {list(interpolation_methods.keys())}"
        )

    series = collection_to_series(collection)
    annual_hourly_index = pd.date_range(
        f"{series.index[0].year}-01-01", periods=8760, freq="h"
    )
    series_annual = series.reindex(annual_hourly_index)
    series_annual[series_annual.index[-1]] = series_annual[series_annual.index[0]]

    return HourlyContinuousCollection(
        header=collection.header,
        values=series_annual.interpolate(method=interpolation_methods[method]).values,
    )


@bhom_analytics()
def peak_time(collection: BaseCollection) -> tuple[Any, tuple[DateTime]]:
    """Find the peak value within a collection, and the time, or times at which it occurs.

    Args:
        collection (BaseCollection):
            A Ladybug DataCollection.

    Returns:
        peak_value, times (tuple[Any, tuple[DateTime]]):
            The peak value and times it occurs.
    """

    peak_value = collection.max
    times = []
    for dt, v in list(zip(*[collection.datetimes, collection.values])):
        if v == peak_value:
            times.append(dt)

    return peak_value, times


@bhom_analytics()
def create_typical_day(
    collection: HourlyContinuousCollection,
    centroid: DateTime,
    sample_size: int,
    agg: str = "mean",
) -> HourlyContinuousCollection:
    """Create a single day representative of conditions centered about the centroid DateTime.

    Args:
        collection (HourlyContinuousCollection):
            The collection to sample from.
        centroid (DateTime):
            The center date around which samples will be taken.
            The centroid will be included in the sample and more granular
            detail than "Date" will be ignored.
        sample_size (int):
            The number of days about the centroid to sample from.
            If even, then this number will be increased by one to ensure
            that half-days are not sampled. A 10 day sample would include
            5 days before and 5 days after the centroid.
        agg (str, optional):
            The aggregation method to use. Defaults to "mean".

    Returns:
        HourlyContinuousCollection:
            The filtered sample. This will have the same header as the input
            collection but will be only 1-day long.
    """

    # check that the collection is hourly
    if not isinstance(collection, HourlyContinuousCollection):
        raise ValueError("The input collection must be hourly.")

    if len(collection.header.analysis_period) < 8760:
        raise ValueError("The input collection must be 1-year long.")

    if collection.header.analysis_period.is_leap_year:
        raise ValueError("The input collection must be non-leap year.")

    ap = AnalysisPeriod(
        st_month=centroid.month,
        st_day=centroid.day,
        end_month=centroid.month,
        end_day=centroid.day,
    )

    if sample_size == 1:
        return collection.filter_by_analysis_period(analysis_period=ap)

    if sample_size % 2 == 0:
        sample_size += 1

    # convert collection to a series over several years
    series = collection_to_series(collection)
    idx = series.index
    serieses = pd.concat([series, series, series], axis=0)
    new_idx = []
    for i in range(-1, 2, 1):
        for ts in idx:
            new_idx.append(
                datetime(
                    year=ts.year + i,
                    month=ts.month,
                    day=ts.day,
                    hour=ts.hour,
                    minute=ts.minute,
                    second=ts.second,
                )
            )
    serieses.index = new_idx

    # get the start and end dates of the sample
    center_date = DateTime(month=centroid.month, day=centroid.day)
    start_date = center_date - timedelta(days=sample_size / 2)
    end_date = center_date + timedelta(days=sample_size / 2)
    sample = serieses.loc[start_date:end_date]
    _base_col = collection.filter_by_analysis_period(
        analysis_period=AnalysisPeriod(
            st_month=centroid.month,
            st_day=centroid.day,
            end_month=centroid.month,
            end_day=centroid.day,
        )
    )
    # pylint: disable=protected-access
    _base_col._values = sample.groupby(sample.index.hour).agg(agg).values
    # pylint: enable=protected-access
    return _base_col
