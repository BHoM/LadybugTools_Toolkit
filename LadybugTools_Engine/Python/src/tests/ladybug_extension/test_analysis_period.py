import random
from datetime import datetime, timedelta

import pytest
from ladybug.analysisperiod import AnalysisPeriod

from ...ladybugtools_toolkit.ladybug_extension.analysis_period import (
    describe,
    from_datetimes,
    to_datetimes,
)


def test_from_datetimes():
    """_"""
    datetimes = [
        datetime(2007, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(8760)
    ]
    assert from_datetimes(datetimes).is_annual


def test_from_datetimes_bad():
    """_"""
    datetimes = random.sample(
        [datetime(2007, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(8760)], 1200
    )
    with pytest.raises((ValueError, ZeroDivisionError)):
        from_datetimes(datetimes)


def test_to_datetimes():
    """_"""
    assert (
        len(
            to_datetimes(
                AnalysisPeriod(st_month=3, end_month=6, st_hour=6, end_hour=12)
            )
        )
        == 854
    )


def test_describe():
    """_"""
    assert describe(AnalysisPeriod()) == "Jan 01 to Dec 31 between 00:00 and 00:00"
    assert (
        describe(AnalysisPeriod(timestep=30), include_timestep=True)
        == "Jan 01 to Dec 31 between 00:00 and 00:00, every 2 minutes"
    )
    assert (
        describe(
            AnalysisPeriod(
                timestep=30, st_month=11, end_month=2, st_hour=18, end_hour=13
            ),
            include_timestep=True,
        )
        == "Nov 01 to Feb 28 between 18:00 and 14:00, every 2 minutes"
    )
    assert (
        describe(
            AnalysisPeriod(
                timestep=2, st_month=11, end_month=2, st_hour=18, end_hour=13
            ),
            include_timestep=True,
            save_path=True,
        )
        == "1101_0228_18_13_02"
    )
