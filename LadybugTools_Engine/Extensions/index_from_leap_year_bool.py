import pandas as pd


def index_from_leap_year_bool(is_leap_year: bool) -> pd.DatetimeIndex:
    """Generate a normal year pandas DatetimeIndex.

    Args:
        is_leap_year (bool): True if a leap year index should be generated, containing 8784 values.

    Returns:
        DatetimeIndex: A pandas DatetimeIndex object.
    """
    n_hours = 8784 if is_leap_year else 8760
    year = 2020 if is_leap_year else 2021
    return pd.date_range(
        f"{year}-01-01 00:30:00", freq="60T", periods=n_hours, name="timestamp"
    )
