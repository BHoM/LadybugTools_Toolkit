"""Methods for manipulating Ladybug datetime objects."""

from datetime import datetime  # pylint: disable=E0401

from ladybug.dt import DateTime
from ..bhom import decorator_factory


@decorator_factory()
def lb_datetime_to_datetime(lb_datetime: DateTime) -> datetime:
    """Convert a Ladybug DateTime object into a Python datetime object.

    Args:
        lb_datetime (DateTime):
            A Ladybug DateTime object.

    Returns:
        datetime:
            A Python datetime object.
    """
    return datetime(
        year=lb_datetime.year,
        month=lb_datetime.month,
        day=lb_datetime.day,
        hour=lb_datetime.hour,
        minute=lb_datetime.minute,
        second=lb_datetime.second,
    )


@decorator_factory()
def lb_datetime_from_datetime(date_time: datetime) -> DateTime:
    """Convert a Python datetime object into a Ladybug DateTime object.

    Args:
        date_time (datetime):
            A Python datetime object.

    Returns:
        DateTime:
            A Ladybug DateTime object.
    """

    leap_year = (date_time.year % 4 == 0 and date_time.year % 100 != 0) or (
        date_time.year % 400 == 0
    )

    return DateTime(
        month=date_time.month,
        day=date_time.day,
        hour=date_time.hour,
        minute=date_time.minute,
        leap_year=leap_year,
    )
