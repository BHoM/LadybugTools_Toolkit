import pandas as pd
import numpy as np
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.generic import GenericType
from ladybug.datacollection import HourlyContinuousCollection

from .solar_time_hour import solar_time_hour as sth
from ..analysis_period import to_datetimes


def solar_time_datetime(
    epw: EPW, solar_time_hour: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (as datetime) for each hour of the year.

    Args:
        epw (EPW): An EPW object.
        solar_time_hour (HourlyContinuousCollection, optional): A pre-calculated solar time (hour) HourlyContinuousCollection. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar times as datetime objects.
    """

    if solar_time_hour is None:
        solar_time_hour = sth(epw)

    timestamp_str = [
        f"{int(i):02d}:{int(np.floor((i*60) % 60)):02d}:{(i*3600) % 60:0.8f}"
        for i in solar_time_hour
    ]
    date_str = to_datetimes(epw.dry_bulb_temperature).strftime("%Y-%m-%d")
    _datetimes = pd.to_datetime(
        [f"{ds} {ts}" for ds, ts in list(zip(*[date_str, timestamp_str]))]
    )
    _datetimes = list(_datetimes)

    # Sometimes the first datetime for solar time occurs before the target year - so this moves the first datetime to the previous day
    for i in range(12):
        if (_datetimes[i].year == _datetimes[-1].year) and (_datetimes[i].hour > 12):
            _datetimes[i] = _datetimes[i] - pd.Timedelta(days=1)

    return HourlyContinuousCollection(
        Header(
            data_type=GenericType(
                name="Solar Time",
                unit="datetime",
            ),
            unit="datetime",
            analysis_period=AnalysisPeriod(),
            metadata=solar_time_hour.header.metadata,
        ),
        _datetimes,
    )
