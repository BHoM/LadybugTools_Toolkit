import warnings
from typing import List

from ladybug._datacollectionbase import BaseCollection
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.angle import Angle
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.analysis_period.to_datetimes import (
    to_datetimes,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


from ladybugtools_toolkit import analytics


@analytics
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
