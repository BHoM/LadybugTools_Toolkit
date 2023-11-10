from datetime import datetime

from ladybug.dt import DateTime
from ladybugtools_toolkit.ladybug_extension.dt import (
    lb_datetime_from_datetime,
    lb_datetime_to_datetime,
)


def test_from_datetime():
    """_"""
    assert lb_datetime_from_datetime(datetime(2007, 1, 1, 1, 30, 0)).hoy == 1.5


def test_to_datetime():
    """_"""
    assert (
        lb_datetime_to_datetime(DateTime(month=1, day=1, hour=12, minute=0)).hour == 12
    )
