from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import datetime
import itertools
import multiprocessing
from typing import List

import numpy as np
import pandas as pd
from ladybug import datatype
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.location import Location
from ladybug.psychrometrics import (
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    rel_humid_from_db_dpt,
    wet_bulb_from_db_rh,
)
from ladybug.skymodel import clearness_index
from ladybug.sunpath import Sun, Sunpath
from ladybug.wea import Wea

<<<<<<< HEAD
from ladybug_extension.analysis_period import to_datetimes
from ladybug_extension.datacollection import to_hourly, to_series


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
        except (AttributeError, TypeError, ZeroDivisionError, ValueError) as e:
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
    _datetimes = list(_datetimes)

    # Sometimes the first datetime for solar time occurs before the target year - so this moves the first datetime to the previous day
    for i in range(12):
        if (_datetimes[i].year == _datetimes[-1].year) and (_datetimes[i].hour > 12):
            _datetimes[i] = _datetimes[i] - pd.Timedelta(days=1)

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
        _datetimes,
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


def sun_position(epw: EPW) -> List[Sun]:
    """
    Calculate sun positions for a given epw file.

    Args:
        epw (EPW): An epw object.
    Returns:
        List[Sun]: A list of Sun objects.
    """

    sunpath = Sunpath.from_location(epw.location)

    return [sunpath.calculate_sun_from_hoy(i) for i in range(len(epw.years))]


def sun_azimuth(epw: EPW) -> List[float]:
    """
    Calculate sun azimuths for a given epw file.

    Args:
        epw (EPW): An epw object.
    Returns:
        List[float]: A list of sun azimuths.
    """
    suns = sun_position(epw)
    return [sun.azimuth for sun in suns]


def sun_altitude(epw: EPW) -> List[float]:
    """
    Calculate sun altitudes for a given epw file.

    Args:
        epw (EPW): An epw object.
    Returns:
        List[float]: A list of sun altitudes.
    """
    suns = sun_position(epw)
    return [sun.altitude for sun in suns]
=======
from ladybug_extension.datacollection import to_datetimes, to_hourly, to_series
>>>>>>> eb8687a (interim - not working)


def from_dataframe(dataframe: pd.DataFrame, location: Location = None) -> EPW:
    """Create an EPW object from a Pandas DataFrame with named columns.

    Args:
        dataframe (pd.DataFrame): A Pandas DataFrame with named columns.
        location (Location, optional): A ladybug Location object. Defaults to None.

    Returns:
        EPW: An EPW object.
    """

    # Check dataframe shape for leapedness and length
    if dataframe.index.is_leap_year.any():
        leap_yr = True
        assert (
            len(dataframe.index) == 8784
        ), "The dataframe must have 8784 rows for leap years."
    else:
        leap_yr = False
        assert (
            len(dataframe.index) == 8760
        ), "The dataframe must have 8760 rows for non-leap years."

    if location is None:
        location = Location()
    try:
        location.source += "[Custom EPW from Pandas DataFrame]"
    except TypeError:
        location.source = "[Custom EPW from Pandas DataFrame]"

    epw_obj = EPW.from_missing_values(is_leap_year=leap_yr)
    epw_obj.location = location

    # Assign data to the EPW
    try:
        epw_obj.aerosol_optical_depth.values = dataframe[
            "Aerosol Optical Depth (fraction)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.albedo.values = dataframe["Albedo (fraction)"].values
    except KeyError:
        pass

    try:
        epw_obj.atmospheric_station_pressure.values = dataframe[
            "Atmospheric Station Pressure (Pa)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.ceiling_height.values = dataframe["Ceiling Height (m)"].values
    except KeyError:
        pass

    try:
        epw_obj.extraterrestrial_direct_normal_radiation.values = dataframe[
            "Extraterrestrial Direct Normal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.extraterrestrial_horizontal_radiation.values = dataframe[
            "Extraterrestrial Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.liquid_precipitation_depth.values = dataframe[
            "Liquid Precipitation Depth (mm)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.liquid_precipitation_quantity.values = dataframe[
            "Liquid Precipitation Quantity (fraction)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.days_since_last_snowfall.values = dataframe[
            "Days Since Last Snowfall (day)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.dry_bulb_temperature.values = dataframe[
            "Dry Bulb Temperature (C)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.dew_point_temperature.values = dataframe[
            "Dew Point Temperature (C)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.wind_speed.values = dataframe["Wind Speed (m/s)"].values
    except KeyError:
        pass

    try:
        epw_obj.wind_direction.values = dataframe["Wind Direction (degrees)"].values
    except KeyError:
        pass

    try:
        epw_obj.direct_normal_radiation.values = dataframe[
            "Direct Normal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.snow_depth.values = dataframe["Snow Depth (cm)"].values
    except KeyError:
        pass

    try:
        epw_obj.diffuse_horizontal_radiation.values = dataframe[
            "Diffuse Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.horizontal_infrared_radiation_intensity.values = dataframe[
            "Horizontal Infrared Radiation Intensity (W/m2)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.direct_normal_illuminance.values = dataframe[
            "Direct Normal Illuminance (lux)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.diffuse_horizontal_illuminance.values = dataframe[
            "Diffuse Horizontal Illuminance (lux)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.precipitable_water.values = dataframe["Precipitable Water (mm)"].values
    except KeyError:
        pass

    try:
        epw_obj.present_weather_codes.values = dataframe[
            "Present Weather Codes (codes)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.present_weather_observation.values = dataframe[
            "Present Weather Observation (observation)"
        ].values
    except KeyError:
        pass

    try:
        epw_obj.total_sky_cover.values = dataframe["Total Sky Cover (tenths)"].values
    except KeyError:
        pass

    try:
        epw_obj.opaque_sky_cover.values = dataframe["Opqaue Sky Cover (tenths)"].values
    except KeyError:
        pass

    try:
        epw_obj.visibility.values = dataframe["Visibility (km)"].values
    except KeyError:
        pass

    try:
        epw_obj.zenith_luminance.values = dataframe["Zenith Luminance (cd/m2)"].values
    except KeyError:
        pass

    try:
        epw_obj.relative_humidity.values = dataframe["Relative Humidity (%)"].values
    except KeyError:
        epw_obj.relative_humidity.values = (
            HourlyContinuousCollection.compute_function_aligned(
                rel_humid_from_db_dpt,
                [epw_obj.dry_bulb_temperature, epw_obj.dew_point_temperature],
                datatype.fraction.RelativeHumidity(),
                "%",
            ).values
        )

    try:
        epw_obj.global_horizontal_radiation.values = dataframe[
            "Global Horizontal Radiation (Wh/m2)"
        ].values
    except KeyError:
        wea = Wea(
            location,
            epw_obj.direct_normal_radiation,
            epw_obj.diffuse_horizontal_radiation,
        )
        epw_obj.global_horizontal_radiation.values = (
            wea.global_horizontal_irradiance.values
        )

    try:
        epw_obj.global_horizontal_illuminance.values = dataframe[
            "Global Horizontal Illuminance (lux)"
        ].values
    except KeyError:
        glob_horiz = []
        sp = Sunpath.from_location(location)
        sp.is_leap_year = leap_yr
        for dt, dni, dhi in zip(
            epw_obj.direct_normal_illuminance.datetimes,
            epw_obj.direct_normal_illuminance,
            epw_obj.diffuse_horizontal_illuminance,
        ):
            sun = sp.calculate_sun_from_date_time(dt)
            glob_horiz.append(dhi + dni * np.sin(np.radians(sun.altitude)))
        epw_obj.global_horizontal_illuminance.values = glob_horiz

    return epw_obj


def _epw_equality(epw0: EPW, epw1: EPW, include_header: bool = False) -> bool:
    """Check for equality between two EPW objects, with regards to the data contained within.

    Args:
        epw0 (EPW): A ladybug EPW object.
        epw1 (EPW): A ladybug EPW object.
        include_header (bool, optional): Include the str repsresentation of the EPW files header in the comparison. Defaults to False.

    Returns:
        bool: True if the two EPW objects are equal, False otherwise.
    """
    if not isinstance(epw0, EPW) or not isinstance(epw1, EPW):
        raise TypeError("Both inputs must be of type EPW.")

    if include_header:
        if epw0.header != epw1.header:
            return False

    # Check key metrics
    dbt_match = epw0.dry_bulb_temperature == epw1.dry_bulb_temperature
    rh_match = epw0.relative_humidity == epw1.relative_humidity
    dpt_match = epw0.dew_point_temperature == epw1.dew_point_temperature
    ws_match = epw0.wind_speed == epw1.wind_speed
    wd_match = epw0.wind_direction == epw1.wind_direction
    ghr_match = epw0.global_horizontal_radiation == epw1.global_horizontal_radiation
    dnr_match = epw0.direct_normal_radiation == epw1.direct_normal_radiation
    dhr_match = epw0.diffuse_horizontal_radiation == epw1.diffuse_horizontal_radiation
    atm_match = epw0.atmospheric_station_pressure == epw1.atmospheric_station_pressure

    return all(
        [
            dbt_match,
            rh_match,
            dpt_match,
            ws_match,
            wd_match,
            ghr_match,
            dnr_match,
            dhr_match,
            atm_match,
        ]
    )


def radiation_tilt_orientation_matrix(epw: EPW) -> pd.DataFrame:
    """Compute the annual cumulative radiation matrix per surface tilt and orientation, for a given EPW object.

    Args:
        epw (EPW): The EPW object for which this calculation is made.

    Returns:
        pd.DataFrame: _description_
    """    
    
    wea = Wea.from_annual_values(epw.location, epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values, is_leap_year=epw.is_leap_year)
    
    # I implement a bit of a hack here, to calculate only the Eastern insolation - then mirror it about the North-South axis to get the whole matrix
    altitudes = np.linspace(0, 90, 10)
    azimuths = np.linspace(0, 180, 19)
    combinations = np.array(list(itertools.product(altitudes, azimuths)))

    def f(alt_az):
        return wea.__copy__().directional_irradiance(alt_az[0], alt_az[1])[0].total
    
    with ThreadPoolExecutor() as executor:
        results = np.array([i for i in executor.map(f, combinations[0:])]).reshape(len(altitudes), len(azimuths))

    temp = pd.DataFrame(results, index=altitudes, columns=azimuths)

    new_cols = (360 - temp.columns)[::-1][1:]
    new_vals = temp.values[::-1, ::-1][::-1, 1:]  # some weird array transformation stuff here
    mirrored = pd.DataFrame(new_vals, columns=new_cols, index=temp.index)
    return pd.concat([temp, mirrored], axis=1)
