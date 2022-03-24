import datetime
import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import numpy as np
import pandas as pd
from ladybug import datatype
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.psychrometrics import (
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    wet_bulb_from_db_rh,
)
from ladybug.skymodel import clearness_index
from ladybug.sunpath import Sunpath
from ladybug_extension.analysisperiod.to_datetimes import to_datetimes
from ladybug_extension.datacollection.to_hourly import to_hourly
from ladybug_extension.datacollection.to_series import to_series
from ladybug_extension.epw.sun_position import sun_position


def to_dataframe(
    epw: EPW,
) -> pd.DataFrame:
    """Create a Pandas DataFrame from an EPW object, including additional calculated properties.

    Args:
        epw (EPW): An EPW object.

    Returns:
        pd.DataFrame: A Pandas DataFrame with the EPW data and additional calculated properties.
    """

    all_series = []
    for p in dir(epw):
        try:
            all_series.append(to_series(getattr(epw, p)))
        except (AttributeError, TypeError, ZeroDivisionError) as e:
            pass

    for k, v in epw.monthly_ground_temperature.items():
        hourly_collection = to_hourly(v)
        hourly_series = to_series(hourly_collection)
        hourly_series.name = f"{hourly_series.name} at {k}m"
        all_series.append(hourly_series)

    # Calculate additional solar properties
    sun_position = _get_sun_position(epw)
    equation_of_time = _get_equation_of_time(epw)
    solar_time_hour = _get_solar_time_hour(epw, equation_of_time)
    solar_altitude = _get_solar_altitude(epw, sun_position)
    solar_altitude_in_radians = _get_solar_altitude_in_radians(epw, sun_position)
    solar_declination = _get_solar_declination(epw)
    solar_time_datetime = _get_solar_time_datetime(epw, solar_time_hour)
    solar_azimuth = _get_solar_azimuth(epw, sun_position)
    solar_azimuth_in_radians = _get_solar_azimuth_in_radians(epw, sun_position)
    apparent_solar_zenith = _get_apparent_solar_zenith(epw, solar_altitude)
    clearness_index = get_clearness_index(epw, sun_position)

    # Calculate additional psychrometric properties
    humidity_ratio = get_humidity_ratio(epw)
    enthalpy = get_enthalpy(epw, humidity_ratio)
    wet_bulb_temperature = get_wet_bulb_temperature(epw)

    for collection in [
        equation_of_time,
        solar_time_hour,
        solar_altitude,
        solar_declination,
        solar_time_datetime,
        solar_azimuth,
        solar_azimuth_in_radians,
        apparent_solar_zenith,
        solar_altitude,
        solar_altitude_in_radians,
        humidity_ratio,
        enthalpy,
        wet_bulb_temperature,
        clearness_index,
    ]:
        all_series.append(to_series(collection))

    # Compile all the data into a dataframe
    df = pd.concat(all_series, axis=1)  # .sort_index(axis=1)
    # df.columns = pd.MultiIndex.from_tuples(df.columns, names=('variable', 'unit', 'location'))

    return df


def _get_sun_position(epw: EPW) -> HourlyContinuousCollection:
    """Calculate a set of Sun positions for each hour of the year.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of sun positions.
    """

    suns = sun_position(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.generic.GenericType(name="Sun Position", unit="Sun"),
            unit="Sun",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        suns,
    )


def _get_solar_declination(epw: EPW) -> HourlyContinuousCollection:
    """Calculate solar declination for each hour of the year.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar declinations.
    """
    sunpath = Sunpath.from_location(epw.location)

    solar_declination_values, _ = list(
        zip(
            *[
                list(sunpath._calculate_solar_geometry(i))
                for i in epw.dry_bulb_temperature.datetimes
            ]
        )
    )

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.angle.Angle(name="Solar Declination"),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        solar_declination_values,
    )


def _get_equation_of_time(epw: EPW) -> HourlyContinuousCollection:
    """Calculate the equation of time for each hour of the year.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of equation of times.
    """
    sunpath = Sunpath.from_location(epw.location)

    _, equation_of_time = list(
        zip(
            *[
                list(sunpath._calculate_solar_geometry(i))
                for i in epw.dry_bulb_temperature.datetimes
            ]
        )
    )

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.time.Time(name="Equation of Time"),
            unit="min",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        equation_of_time,
    )


def _get_solar_time_hour(
    epw: EPW, equation_of_time: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (in hour-of-day) for each hour of the year.

    Args:
        epw (EPW): An EPW object.
        equation_of_time (HourlyContinuousCollection, optional): A pre-calculated equation of time HourlyContinuousCollection. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar time (in hour-of-day).
    """

    if equation_of_time is None:
        equation_of_time = _get_equation_of_time(epw)

    sunpath = Sunpath.from_location(epw.location)
    hour_values = [i.hour for i in epw.dry_bulb_temperature.datetimes]

    solar_time = [
        sunpath._calculate_solar_time(j, k, False)
        for j, k in list(zip(*[hour_values, equation_of_time.values]))
    ]

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.time.Time(name="Solar Time"),
            unit="hr",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        solar_time,
    )


def _get_solar_time_datetime(
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
        solar_time_hour = _get_solar_time_hour(epw)

    timestamp_str = [
        f"{int(i):02d}:{int(np.floor((i*60) % 60)):02d}:{(i*3600) % 60:0.8f}"
        for i in solar_time_hour
    ]
    date_str = to_datetimes(epw.dry_bulb_temperature).strftime("%Y-%m-%d")
    _datetimes = pd.to_datetime(
        [f"{ds} {ts}" for ds, ts in list(zip(*[date_str, timestamp_str]))]
    )

    # Sometimes the first datetime for solar time occurs before the target year - so this moves the first datetime to the previous day
    if _datetimes[0] > _datetimes[1]:
        datetimes = [_datetimes[0] - datetime.timedelta(hours=24)] + _datetimes[
            1:
        ].to_list()
    else:
        datetimes = _datetimes

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.generic.GenericType(
                name="Solar Time",
                unit="datetime",
            ),
            unit="datetime",
            analysis_period=AnalysisPeriod(),
            metadata=solar_time_hour.header.metadata,
        ),
        datetimes,
    )


def _get_solar_azimuth(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle.

    Args:
        epw (EPW): An EPW object.
        sun_position (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar azimuth angles.
    """

    if not sun_position:
        sun_position = _get_sun_position(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.angle.Angle(
                name="Solar Azimuth",
            ),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.azimuth for i in sun_position.values],
    )


def _get_solar_azimuth_in_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle in radians.

    Args:
        epw (EPW): An EPW object.
        sun_position (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar azimuth angles in radians.
    """

    if not sun_position:
        sun_position = _get_sun_position(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.angle.Angle(
                name="Solar Azimuth",
            ),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.azimuth_in_radians for i in sun_position.values],
    )


def _get_solar_altitude(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle.

    Args:
        epw (EPW): An EPW object.
        sun_position (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar altitude angles.
    """

    if not sun_position:
        sun_position = _get_sun_position(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.angle.Angle(
                name="Solar Altitude",
            ),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.altitude for i in sun_position.values],
    )


def _get_solar_altitude_in_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle in radians.

    Args:
        epw (EPW): An EPW object.
        sun_position (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of solar altitude angles in radians.
    """

    if not sun_position:
        sun_position = _get_sun_position(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.angle.Angle(
                name="Solar Altitude",
            ),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.altitude_in_radians for i in sun_position.values],
    )


def _get_apparent_solar_zenith(
    epw: EPW, solar_altitude: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly apparent solar zenith angles.

    Args:
        epw (EPW): An EPW object.
        solar_altitude (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of solar altitude angles. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of apparent solar zenith angles.
    """

    if not solar_altitude:
        solar_altitude = _get_solar_altitude(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.angle.Angle(name="Apparent Solar Zenith"),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [np.pi / 2 - i for i in solar_altitude.values],
    )


def get_wet_bulb_temperature(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly wet bulb temperature for a given EPW.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of wet bulb temperatures.
    """
    return HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        datatype.temperature.WetBulbTemperature(),
        "C",
    )


def get_humidity_ratio(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly humidity ratio for a given EPW.

    Args:
        epw (EPW): An EPW object.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of humidity ratios.
    """
    return HourlyContinuousCollection.compute_function_aligned(
        humid_ratio_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        datatype.fraction.HumidityRatio(),
        "fraction",
    )


def get_enthalpy(
    epw: EPW, humidity_ratio: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate an annual hourly enthalpy for a given EPW.

    Args:
        epw (EPW): _description_
        humidity_ratio (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of humidity ratios. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of enthalpies.
    """

    if not humidity_ratio:
        humidity_ratio = get_humidity_ratio(epw)

    return HourlyContinuousCollection.compute_function_aligned(
        enthalpy_from_db_hr,
        [
            epw.dry_bulb_temperature,
            humidity_ratio,
        ],
        datatype.specificenergy.Enthalpy(),
        "kJ/kg",
    )


def get_clearness_index(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate the clearness index value for each hour of the year.

    Args:
        epw (EPW): An EPW object.
        sun_position (HourlyContinuousCollection, optional): A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection: An HourlyContinuousCollection of clearness indices.
    """

    if not sun_position:
        sun_position = _get_sun_position(epw)

    ci = []
    for i, j, k in list(
        zip(
            *[
                epw.global_horizontal_radiation,
                _get_solar_altitude(epw, sun_position),
                epw.extraterrestrial_direct_normal_radiation,
            ]
        )
    ):
        try:
            ci.append(clearness_index(i, j, k))
        except ZeroDivisionError:
            ci.append(0)

    return HourlyContinuousCollection(
        header=Header(
            data_type=datatype.fraction.Fraction(name="Clearness Index"),
            unit="fraction",
            analysis_period=AnalysisPeriod(),
        ),
        values=ci,
    )
