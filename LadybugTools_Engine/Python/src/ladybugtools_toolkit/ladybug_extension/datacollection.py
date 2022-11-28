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
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    include_time_period: bool = True,
    include_most_common: bool = True,
    include_total: bool = True,
    include_average: bool = True,
    include_median: bool = True,
    include_min: bool = True,
    include_max: bool = True,
    include_lowest_month: bool = True,
    include_highest_month: bool = True,
    include_lowest_monthly_diurnal: bool = True,
    include_highest_monthly_diurnal: bool = True,
    include_max_time: bool = True,
    include_min_time: bool = True,
) -> List[str]:
    """Describe a datacollection.

    Args:
        ...

    Returns:
        List[str]: A list of strings describing this data collection.

    """

    descriptions = []

    series = to_series(collection).reindex(to_datetimes(analysis_period)).interpolate()

    if isinstance(collection.header.data_type, Angle):
        warnings.warn(
            'Non-hourly "angle" data may be spurious as interpolation around North can result in southerly values.'
        )

    name = " ".join(series.name.split(" ")[:-1])
    unit = series.name.split(" ")[-1].replace("(", "").replace(")", "")

    if include_time_period:
        descriptions.append(
            f"Data collection represents {name}, for {describe_analysis_period(analysis_period, include_timestep=True)}."
        )

    if include_min:
        _min = series.min()
        if _min == collection.header.data_type.min:
            warnings.warn(
                "The minimum value matches the minimum possible value for this collection type."
            )
        _min_idx = series.idxmin()
        descriptions.append(
            f"The minimum {name} is {_min:0.01f}{unit}, occurring on {_min_idx:%b %d} at {_min_idx:%H:%M}."
        )

    if include_max:
        _max = series.max()
        _max_idx = series.idxmax()
        descriptions.append(
            f"The maximum {name} is {_max:0.01f}{unit}, occurring on {_max_idx:%b %d} at {_max_idx:%H:%M}."
        )

    if include_average:
        _avg = series.mean()
        _std = series.std()
        descriptions.append(
            f"The average {name} is {_avg:0.01f}{unit}, with a standard deviation of {_std:0.1f}{unit}."
        )

    if include_median:
        _median = series.median()
        _median_count = series[series == _median].count()
        _ = f"The median {name} is {_median:0.01f}{unit}"
        if _median_count > 1:
            _ += f" occurring {_median_count} times."
        else:
            _ += "."
        descriptions.append(_)

    if include_most_common:
        _n_common = 3
        _most_common, _most_common_counts = series.value_counts().reset_index().values.T
        _most_common_str = " and ".join(
            ", ".join(_most_common[:_n_common].astype(str)).rsplit(", ", 1)
        )
        _most_common_counts_str = " and ".join(
            ", ".join(_most_common_counts[:_n_common].astype(int).astype(str)).rsplit(
                ", ", 1
            )
        )
        descriptions.append(
            f"The {_n_common} most common {name} values are {_most_common_str}{unit}, occuring {_most_common_counts_str} times (out of {series.count()}) respectively."
        )

    if include_total:
        _sum = series.sum()
        descriptions.append(f"The cumulative {name} is {_sum:0.1f}{unit}.")

    _month_avg = series.resample("MS").mean()

    if include_lowest_month:
        _min_month_avg = _month_avg.min()
        _min_month = _month_avg.idxmin()
        descriptions.append(
            f"The month with the lowest average {name} is {_min_month:%B}, with an average value of {_min_month_avg:0.1f}{unit}."
        )

    if include_highest_month:
        _max_month_avg = _month_avg.max()
        _max_month = _month_avg.idxmax()
        descriptions.append(
            f"The month with the highest average {name} is {_max_month:%B}, with an average value of {_max_month_avg:0.1f}{unit}."
        )

    _day_resample = series.resample("1D")
    _day_min = _day_resample.min()
    _day_max = _day_resample.max()
    _day_range_month_avg = (_day_max - _day_min).resample("MS").mean().sort_values()

    if include_lowest_monthly_diurnal:
        (
            _diurnal_low_range_month,
            _diurnal_low_range_value,
        ) = _day_range_month_avg.reset_index().values[0]
        descriptions.append(
            f"The month with the lowest average diurnal {name} range is {_diurnal_low_range_month:%B}, with values varying by around {_diurnal_low_range_value:0.1f}{unit}."
        )

    if include_highest_monthly_diurnal:
        (
            _diurnal_high_range_month,
            _diurnal_high_range_value,
        ) = _day_range_month_avg.reset_index().values[-1]
        descriptions.append(
            f"The month with the highest average diurnal {name} range is {_diurnal_high_range_month:%B}, with values varying by around {_diurnal_high_range_value:0.1f}{unit}."
        )

    if include_max_time:
        _max_daily_time = series.groupby(series.index.time).max().idxmax()
        descriptions.append(
            f"During the day, the time where the maximum {name} occurs is usually {_max_daily_time:%H:%M}."
        )

    if include_min_time:
        _min_daily_time = series.groupby(series.index.time).min().idxmin()
        descriptions.append(
            f"During the day, the time where the minimum {name} occurs is usually {_min_daily_time:%H:%M}."
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
