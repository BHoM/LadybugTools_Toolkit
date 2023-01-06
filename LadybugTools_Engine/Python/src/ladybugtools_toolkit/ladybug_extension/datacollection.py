import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    BaseCollection,
    HourlyContinuousCollection,
    MonthlyCollection,
)
from ladybug.datatype.angle import Angle
from ladybug.datautil import (
    collections_from_csv,
    collections_from_json,
    collections_to_csv,
    collections_to_json,
)

from ..helpers import wind_direction_average
from .analysis_period import describe as describe_analysis_period
from .analysis_period import to_datetimes
from .header import from_string as header_from_string
from .header import to_string as header_to_string


def to_series(collection: BaseCollection) -> pd.Series:
    """Convert a Ladybug hourlyContinuousCollection object into a Pandas Series object.

    Args:
        collection: Ladybug data collection object.

    Returns:
        pd.Series: A Pandas Series object.
    """

    index = to_datetimes(collection.header.analysis_period)
    if len(collection.values) == 12:
        index = pd.date_range(f"{index[0].year}-01-01", periods=12, freq="MS")

    return pd.Series(
        data=collection.values,
        index=index,
        name=header_to_string(collection.header),
    )


def to_json(
    collections: List[BaseCollection], json_path: Union[Path, str], indent: int = None
) -> Path:
    """Save Ladybug BaseCollection-like objects into a JSON file.

    Args:
        collection (BaseCollection): A Ladybug BaseCollection-like object.
        json_path (Union[Path, str]): The path to the JSON file.
        indent (str, optional): The indentation to use in the resulting JSON file. Defaults to None.

    Returns:
        Path: The path to the JSON file.

    """

    json_path = Path(json_path)

    if not all(isinstance(n, BaseCollection) for n in collections):
        raise ValueError(
            'All elements of the input "collections" must inherit from BaseCollection.'
        )

    if json_path.suffix != ".json":
        raise ValueError("The target file must be a *.json file.")

    return Path(
        collections_to_json(
            collections,
            folder=json_path.parent.as_posix(),
            file_name=json_path.name,
            indent=indent,
        )
    )


def to_csv(collections: List[BaseCollection], csv_path: Union[Path, str]) -> Path:
    """Save Ladybug BaseCollection-like objects into a CSV file.

    Args:
        collection (BaseCollection): A Ladybug BaseCollection-like object.
        csv_path (Union[Path, str]): The path to the CSV file.

    Returns:
        Path: The path to the CSV file.

    """

    csv_path = Path(csv_path)

    if not all(isinstance(n, BaseCollection) for n in collections):
        raise ValueError(
            'All elements of the input "collections" must inherit from BaseCollection.'
        )

    return Path(
        collections_to_csv(
            collections,
            folder=csv_path.parent.as_posix(),
            file_name=csv_path.name,
        )
    )


def to_array(collection: BaseCollection) -> np.ndarray:
    """Convert a Ladybug BaseCollection-like object into a numpy array.

    Args:
        collection: A Ladybug BaseCollection-like object.

    Returns:
        np.ndarray: A numpy array.
    """

    return np.array(collection.values)


def percentile(
    collections: List[BaseCollection], nth_percentile: float
) -> BaseCollection:
    """Create an nth percentile of the given data collections.

    Args:
        collections (List[BaseCollection]):
            A list of collections.
        percentile (float):
            A percentile, between 0 and 1.

    Returns:
        BaseCollection:
            The "nth percentile" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError(
            'You cannot get the "nth percentile" across non-alike datatypes.'
        )

    # check if any collections input are angular, in which case, fail
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        raise ValueError("This method cannot be applied to Angular datatypes.")

    return from_series(df.quantile(nth_percentile, axis=1).rename(series_name))


def minimum(collections: List[BaseCollection]) -> BaseCollection:
    """Create a Minimum of the given data collections.

    Args:
        collections (List[BaseCollection]):
            A list of collections.

    Returns:
        BaseCollection:
            The "Minimum" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError('You cannot get the "minimum" across non-alike datatypes.')

    # check if any collections input are angular, in which case, fail
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        raise ValueError("This method cannot be applied to Angular datatypes.")
    return from_series(df.min(axis=1).rename(series_name))


def maximum(collections: List[BaseCollection]) -> BaseCollection:
    """Create a Maximum of the given data collections.

    Args:
        collections (List[BaseCollection]):
            A list of collections.

    Returns:
        BaseCollection:
            The "Maximum" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError('You cannot get the "maximum" across non-alike datatypes.')

    # check if any collections input are angular, in which case, fail
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        raise ValueError("This method cannot be applied to Angular datatypes.")
    return from_series(df.max(axis=1).rename(series_name))


def from_series(series: pd.Series) -> BaseCollection:
    """Convert a Pandas Series object into a Ladybug BaseCollection-like object.

    Args:
        series (pd.Series): A Pandas Series object.

    Returns:
        BaseCollection: A Ladybug BaseCollection-like object.
    """

    header = header_from_string(series.name)
    header.metadata["source"] = "From custom pd.Series"

    freq = pd.infer_freq(series.index)
    if freq in ["H"]:
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


def from_json(json_path: Union[Path, str]) -> List[BaseCollection]:
    """Load a JSON containing serialised Ladybug BaseCollection-like objects.

    Args:
        json_path (Union[Path, str]): The path to the JSON file.

    Returns:
        List[BaseCollection]: A list of Ladybug BaseCollection-like object.

    """

    json_path = Path(json_path)

    if json_path.suffix != ".json":
        raise ValueError("The target file must be a *.json file.")

    return collections_from_json(json_path.as_posix())


def from_dict(dictionary: Dict[str, Any]) -> BaseCollection:
    """Convert a JSON compliant dictionary object into a ladybug EPW.

    Args:
        Dict[str, Any]:
            A sanitised dictionary.

    Returns:
        BaseCollection:
            A ladybug collection object.
    """

    json_str = json.dumps(dictionary)

    # custom handling of non-standard JSON NaN/Inf values
    json_str = json_str.replace('"min": "-inf"', '"min": -Infinity')
    json_str = json_str.replace('"max": "inf"', '"max": Infinity')

    try:
        return HourlyContinuousCollection.from_dict(json.loads(json_str))
    except Exception:
        return MonthlyCollection.from_dict(json.loads(json_str))


def from_csv(csv_path: Union[Path, str]) -> List[BaseCollection]:
    """Load a CSV containing serialised Ladybug BaseCollection-like objects.

    Args:
        csv_path (Union[Path, str]): The path to the CSV file.

    Returns:
        List[BaseCollection]: A list of Ladybug BaseCollection-like object.

    """

    csv_path = Path(csv_path)

    if csv_path.suffix != ".csv":
        raise ValueError("The target file must be a *.csv file.")

    return collections_from_csv(csv_path.as_posix())


def describe_collection(
    collection: BaseCollection,
    _n_common: int = 3,
) -> List[str]:
    """Describe a datacollection.

    Args:
        ...

    Returns:
        List[str]: A list of strings describing this data collection.

    """

    descriptions = []

    series = to_series(collection)

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
            f"The data collection being described represents only a portion of a full year."
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
        _mean = wind_direction_average(series.values)
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
            f"Maximum {name} is {_max:,.01f}{unit}, occurring {_n_max} time{f's and typically at {_max_mean_time:%H:%M}' if _n_max > 1 else f' on {series.idxmax():%b %d at %H:%M}'}."
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
        + f" times respectively"
    )
    descriptions.append(
        f"The {_n_common} most common {name} values are {_most_common_str}, occurring {_most_common_counts_str}."
    )

    if not isinstance(collection.header.data_type, Angle):

        # cumulative values
        if rate_of_change:
            descriptions.append(
                f"Cumulative total {name} for {period} is {series.sum():,.0f}{collection.to_time_rate_of_change().header.unit}."
            )

        # agg months
        _month_mean = series.resample("MS").mean()
        descriptions.append(
            f"The month with the lowest mean {name} is {_month_mean.idxmin():%B}, with a value of {_month_mean.min():,.1f}{unit}."
        )
        descriptions.append(
            f"The month with the highest mean {name} is {_month_mean.idxmax():%B}, with a value of {_month_mean.max():,.1f}{unit}."
        )

        # agg times
        _time_mean = series.groupby(series.index.time).mean()
        descriptions.append(
            f"The time when the highest mean {name} typically occurs is {_time_mean.idxmax():%H:%M}, with a value of {_time_mean.max():,.1f}{unit}."
        )
        descriptions.append(
            f"The time when the lowest mean {name} typically occurs is {_time_mean.idxmin():%H:%M}, with a mean value of {_time_mean.min():,.1f}{unit}."
        )

        # diurnal
        _month_grp = series.resample("MS")
        _month_grp_min = _month_grp.min()
        _month_grp_max = _month_grp.max()
        _month_range_month_avg = _month_grp_max - _month_grp_min
        descriptions.append(
            f"The month when the largest range of {name} typically occurs is {_month_range_month_avg.idxmax():%B}, with values between {_month_grp_min.loc[_month_range_month_avg.idxmax()]:,.1f}{unit} and {_month_grp_max.loc[_month_range_month_avg.idxmax()]:,.1f}{unit}."
        )
        descriptions.append(
            f"The month when the smallest range of {name} typically occurs is {_month_range_month_avg.idxmin():%B}, with values between {_month_grp_min.loc[_month_range_month_avg.idxmin()]:,.1f}{unit} and {_month_grp_max.loc[_month_range_month_avg.idxmin()]:,.1f}{unit}."
        )

    return descriptions


def average(collections: List[BaseCollection]) -> BaseCollection:
    """Create an Average of the given data collections.

    Args:
        collections (List[BaseCollection]):
            A list of collections.

    Returns:
        BaseCollection:
            The "Average" collection.
    """

    # check all input collections are of the same underlying datatype
    df = pd.concat([to_series(i) for i in collections], axis=1)

    series_name = df.columns[0]
    if len(np.unique(df.columns)) != 1:
        raise ValueError('You cannot get the "average" across non-alike datatypes.')

    # check if any collections input are angular, in which case, use a different method
    if ("angle" in series_name.lower()) or ("direction" in series_name.lower()):
        # convert to radians if not already there
        if "degree" in series_name.lower():
            df = np.rad2deg(df)
        angles = []
        for _, row in df.iterrows():
            angles.append(
                np.round(
                    np.arctan2(
                        (1 / len(row) * np.sin(row)).sum(),
                        (1 / len(row) * np.cos(row)).sum(),
                    ),
                    2,
                )
            )
        angles = np.array(angles)
        series = pd.Series(
            np.where(angles < 0, (np.pi * 2) - -angles, angles),
            index=df.index,
            name=series_name,
        )
        if "degree" in series_name.lower():
            series = np.rad2deg(series)
    else:
        series = df.mean(axis=1).rename(series_name)

    return from_series(series)


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

    if method not in interpolation_methods:
        raise ValueError(
            f"The up-sampling method must be one of {list(interpolation_methods.keys())}"
        )

    series = to_series(collection)
    annual_hourly_index = pd.date_range(
        f"{series.index[0].year}-01-01", periods=8760, freq="H"
    )
    series_annual = series.reindex(annual_hourly_index)
    series_annual[series_annual.index[-1]] = series_annual[series_annual.index[0]]

    return HourlyContinuousCollection(
        header=collection.header,
        values=series_annual.interpolate(method=interpolation_methods[method]).values,
    )
