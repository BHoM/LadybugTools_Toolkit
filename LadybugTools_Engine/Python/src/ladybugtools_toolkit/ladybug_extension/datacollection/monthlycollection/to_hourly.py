import pandas as pd
from ladybug.datacollection import (HourlyContinuousCollection,
                                    MonthlyCollection)
from ladybugtools_toolkit import analytics
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import \
    to_series


@analytics
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

    if method not in interpolation_methods.keys():
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
