from datetime import datetime

from ladybug.dt import DateTime


def to_datetime(lb_datetime: DateTime) -> datetime:
    """Convert a Ladybug DateTime object into a Python datetime object.

    Args:
        datetime (DateTime): A Ladybug DateTime object.

    Returns:
        datetime: A Python datetime object.
    """
    return datetime(
        year=lb_datetime.year,
        month=lb_datetime.month,
        day=lb_datetime.day,
        hour=lb_datetime.hour,
        minute=lb_datetime.minute,
        second=lb_datetime.second,
    )


def from_datetime(datetime: datetime) -> DateTime:
    """Convert a Python datetime object into a Ladybug DateTime object.

    Args:
        datetime (datetime): A Python datetime object.

    Returns:
        DateTime: A Ladybug DateTime object.
    """

    leap_year = (datetime.year % 4 == 0 and datetime.year % 100 != 0) or (
        datetime.year % 400 == 0
    )

    return DateTime(
        month=datetime.month,
        day=datetime.day,
        hour=datetime.hour,
        minute=datetime.minute,
        leap_year=leap_year,
    )
