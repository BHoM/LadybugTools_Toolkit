from datetime import datetime

from ladybug.dt import DateTime


from ladybugtools_toolkit import analytics


@analytics
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
