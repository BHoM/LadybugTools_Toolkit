import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from datetime import datetime

from ladybug.dt import DateTime


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
