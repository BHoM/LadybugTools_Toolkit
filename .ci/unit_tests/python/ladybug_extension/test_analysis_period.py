import random
from datetime import datetime, timedelta

import pytest
from ladybug.analysisperiod import AnalysisPeriod
from ladybugtools_toolkit.ladybug_extension.analysis_period.from_datetimes import (
    from_datetimes,
)
from ladybugtools_toolkit.ladybug_extension.analysis_period.to_datetimes import (
    to_datetimes,
)


def test_from_datetimes():
    datetimes = [
        datetime(2007, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(8760)
    ]
    assert from_datetimes(datetimes).is_annual


def test_from_datetimes_bad():
    datetimes = random.sample(
        [datetime(2007, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(8760)], 1200
    )
    with pytest.raises(ValueError):
        from_datetimes(datetimes)


def test_to_datetimes():
    assert (
        len(
            to_datetimes(
                AnalysisPeriod(st_month=3, end_month=6, st_hour=6, end_hour=12)
            )
        )
        == 854
    )
