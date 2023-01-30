import calendar
import copy
import datetime
import itertools
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.angle import Angle
from ladybug.datatype.fraction import Fraction, HumidityRatio
from ladybug.datatype.generic import GenericType
from ladybug.datatype.specificenergy import Enthalpy
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.datatype.time import Time
from ladybug.epw import EPW, EPWFields, MonthlyCollection
from ladybug.header import Header
from ladybug.location import Location
from ladybug.psychrometrics import (
    enthalpy_from_db_hr,
    humid_ratio_from_db_rh,
    wet_bulb_from_db_rh,
)
from ladybug.skymodel import clearness_index as lb_ci
from ladybug.sunpath import Sun, Sunpath
from ladybug.wea import Wea
from ladybug_comfort.degreetime import cooling_degree_time, heating_degree_time

from .analysis_period import to_datetimes
from .datacollection import to_series
from .header import to_string as header_to_string


def to_dataframe(epw: EPW, include_additional: bool = False) -> pd.DataFrame:
    """Create a Pandas DataFrame from an EPW object, with option for including additional metrics.

    Args:
        epw (EPW):
            An EPW object.
        include_additional (bool, optional):
            Set to False to not include additional calculated properties. Default is False.

    Returns:
        pd.DataFrame:
            A Pandas DataFrame containing the source EPW data.
    """

    properties = [
        "aerosol_optical_depth",
        "albedo",
        "atmospheric_station_pressure",
        "ceiling_height",
        "days_since_last_snowfall",
        "dew_point_temperature",
        "diffuse_horizontal_illuminance",
        "diffuse_horizontal_radiation",
        "direct_normal_illuminance",
        "direct_normal_radiation",
        "dry_bulb_temperature",
        "extraterrestrial_direct_normal_radiation",
        "extraterrestrial_horizontal_radiation",
        "global_horizontal_illuminance",
        "global_horizontal_radiation",
        "horizontal_infrared_radiation_intensity",
        "liquid_precipitation_depth",
        "liquid_precipitation_quantity",
        "opaque_sky_cover",
        "precipitable_water",
        "present_weather_codes",
        "present_weather_observation",
        "relative_humidity",
        "snow_depth",
        "total_sky_cover",
        "visibility",
        "wind_direction",
        "wind_speed",
        "years",
        "zenith_luminance",
    ]

    all_series = []
    for prop in properties:
        try:
            s = to_series(getattr(epw, prop))
            s.rename((Path(epw.file_path).stem, "EPW", s.name), inplace=True)
            all_series.append(s)
        except ValueError:
            warnings.warn(
                f"{prop} is not available in this EPW file. This is most likely because this file does not follow normal EPW content conventions."
            )

    if not include_additional:
        return pd.concat(all_series, axis=1).sort_index(axis=1)

    # Calculate additional solar properties
    sun_position = sun_position_collection(epw)
    eot = equation_of_time(epw)
    sth = solar_time_hour(epw, eot)
    salt = solar_altitude(epw, sun_position)
    saltr = solar_altitude_radians(epw, sun_position)
    sd = solar_declination(epw)
    stdt = solar_time_datetime(epw, sth)
    saz = solar_azimuth(epw, sun_position)
    sazr = solar_azimuth_radians(epw, sun_position)
    ci = clearness_index(epw, sun_position)

    # Calculate additional psychrometric properties
    hr = humidity_ratio(epw)
    ent = enthalpy(epw, hr)
    wbt = wet_bulb_temperature(epw)

    # Add properties to DataFrame
    for collection in [
        eot,
        sth,
        salt,
        saltr,
        sd,
        stdt,
        saz,
        sazr,
        ci,
        hr,
        ent,
        wbt,
    ]:
        s = to_series(collection)
        s.rename((Path(epw.file_path).stem, "EPW", s.name), inplace=True)
        all_series.append(s)

    return pd.concat(all_series, axis=1).sort_index(axis=1)


def from_dataframe(
    dataframe: pd.DataFrame,
    location: Location = None,
    monthly_ground_temperature: Dict[float, MonthlyCollection] = None,
    comments_1: str = None,
    comments_2: str = None,
) -> EPW:
    """Create an EPW object from a Pandas DataFrame with named columns.

    Args:
        dataframe (pd.DataFrame):
            A Pandas DataFrame with named columns in the format created by the to_dataframe method.
        location (Location, optional):
            A ladybug Location object. Defaults to None which results in a default being applied.
        monthly_ground_temperature (Dict[float, MonthlyCollection], optional):
            A dictionary of monthly ground temperatures. Default is None.
        comments_1 (str, optional):
            A string to be added as comment to the resultant object. Default is None.
        comments_2 (str, optional):
            Another string to be added as comment to the resultant object. Default is None.

    Returns:
        EPW:
            An EPW object.
    """

    # Check dataframe shape for leaped-ness and length
    if sum((dataframe.index.month == 2) & (dataframe.index.day == 29)) != 0:
        leap_yr = True
        if len(dataframe.index) != 8784:
            raise ValueError(
                "The dataframe must have 8784 rows as it contains a 29th of February, suggesting a leap year."
            )
    else:
        leap_yr = False
        if len(dataframe.index) != 8760:
            raise ValueError(
                "The dataframe must have 8760 rows as it does not contain a 29th of February, suggesting a non-leap year."
            )

    # adjust columns format to match expected format
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(-1)
    elif not isinstance(dataframe.columns[0], str):
        raise ValueError(
            "The dataframes column headers are not in the expected format."
        )

    # create "empty" EPW object
    epw_obj = EPW.from_missing_values(is_leap_year=leap_yr)

    # Add "location" attributes
    if location is None:
        location = Location()
    location = copy.copy(location)
    try:
        location.source += "[Custom EPW from Pandas DataFrame]"
    except TypeError:
        location.source = "[Custom EPW from Pandas DataFrame]"
    epw_obj.location = location

    # Add ground temperatures if available
    if monthly_ground_temperature:
        epw_obj.monthly_ground_temperature = monthly_ground_temperature

    # Add comments if provided
    if comments_1:
        epw_obj.comments_1 = comments_1
    if comments_2:
        epw_obj.comments_2 = comments_2

    # Assign data to the EPW
    properties = [
        "aerosol_optical_depth",
        "albedo",
        "atmospheric_station_pressure",
        "ceiling_height",
        "days_since_last_snowfall",
        "dew_point_temperature",
        "diffuse_horizontal_illuminance",
        "diffuse_horizontal_radiation",
        "direct_normal_illuminance",
        "direct_normal_radiation",
        "dry_bulb_temperature",
        "extraterrestrial_direct_normal_radiation",
        "extraterrestrial_horizontal_radiation",
        "global_horizontal_illuminance",
        "global_horizontal_radiation",
        "horizontal_infrared_radiation_intensity",
        "liquid_precipitation_depth",
        "liquid_precipitation_quantity",
        "opaque_sky_cover",
        "precipitable_water",
        "present_weather_codes",
        "present_weather_observation",
        "relative_humidity",
        "snow_depth",
        "total_sky_cover",
        "visibility",
        "wind_direction",
        "wind_speed",
        "years",
        "zenith_luminance",
    ]

    try:
        for prop in properties:
            setattr(
                getattr(epw_obj, prop),
                "values",
                dataframe[header_to_string(getattr(epw_obj, prop).header)].values,
            )
    except KeyError:
        warnings.warn(
            f"{prop} cannot be added to EPW as it doesn't exist in the Pandas DataFrame."
        )

    return epw_obj


def wet_bulb_temperature(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly wet bulb temperature for a given EPW.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of wet bulb temperatures.
    """
    return HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        WetBulbTemperature(),
        "C",
    )


def unique_wind_speed_direction(
    epw: EPW, schedule: List[int] = None
) -> List[Tuple[float, float]]:
    """Return a set of unique wind speeds and directions for an EPW file.

    Args:
        epw (EPW): An epw object.
        schedule (epw): a mask of hours to include in the unique

    Returns:
        List[List[float, float]]: A list of unique (wind_speed, wind_direction).
    """

    df = pd.concat([to_series(epw.wind_speed), to_series(epw.wind_direction)], axis=1)

    if schedule is not None:
        df = df.iloc[schedule]

    return df.drop_duplicates().values


def to_dict(epw: EPW) -> Dict[str, Any]:
    """Convert a ladybug EPW object into a JSON-able compliant dictionary.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        Dict[str, Any]:
            A sanitised dictionary.
    """
    d = epw.to_dict()
    json_str = json.dumps(d)

    # custom handling of non-standard JSON NaN/Inf values
    json_str = json_str.replace('"min": -Infinity', '"min": "-inf"')
    json_str = json_str.replace('"max": Infinity', '"max": "inf"')

    # custom handling of float-indexed values
    for k, _ in d["monthly_ground_temps"].items():
        json_str = json_str.replace(f'"{k}": {{', f'"_{k}": {{'.replace(".", "_"))

    return json.loads(json_str)


def sun_position_list(epw: EPW) -> List[Sun]:
    """
    Calculate sun positions for a given epw file.

    Args:
        epw (EPW):
            An epw object.
    Returns:
        List[Sun]:
            A list of Sun objects.
    """

    sunpath = Sunpath.from_location(epw.location)

    return [sunpath.calculate_sun_from_hoy(i) for i in range(len(epw.years))]


def sun_position_collection(epw: EPW) -> HourlyContinuousCollection:
    """Calculate a set of Sun positions for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of sun positions.
    """

    suns = sun_position_list(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=GenericType(name="Sun Position", unit="Sun"),
            unit="Sun",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        suns,
    )


def solar_time_hour(
    epw: EPW, eot: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (in hour-of-day) for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.
        eot (HourlyContinuousCollection, optional):
            A pre-calculated equation of time HourlyContinuousCollection. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar time (in hour-of-day).
    """

    if eot is None:
        eot = equation_of_time(epw)

    sunpath = Sunpath.from_location(epw.location)
    hour_values = [i.hour for i in epw.dry_bulb_temperature.datetimes]

    solar_time = [
        sunpath._calculate_solar_time(j, k, False)
        for j, k in list(zip(*[hour_values, eot.values]))
    ]

    return HourlyContinuousCollection(
        Header(
            data_type=Time(name="Solar Time"),
            unit="hr",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        solar_time,
    )


def solar_time_datetime(
    epw: EPW, sth: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (as datetime) for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.
        solar_time_hour (HourlyContinuousCollection, optional):
            A pre-calculated solar time (hour) HourlyContinuousCollection. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar times as datetime objects.
    """

    if sth is None:
        sth = solar_time_hour(epw)

    timestamp_str = [
        f"{int(i):02d}:{int(np.floor((i*60) % 60)):02d}:{(i*3600) % 60:0.8f}"
        for i in sth
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
            metadata=sth.header.metadata,
        ),
        _datetimes,
    )


def solar_declination(epw: EPW) -> HourlyContinuousCollection:
    """Calculate solar declination for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar declinations.
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
            data_type=Angle(name="Solar Declination"),
            unit="radians",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        solar_declination_values,
    )


def solar_azimuth(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar azimuth angles.
    """

    if not sun_position:
        sun_position = sun_position_collection(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=Angle(
                name="Solar Azimuth",
            ),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.azimuth for i in sun_position.values],
    )


def solar_azimuth_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar azimuth angles.
    """

    collection = solar_azimuth(epw, sun_position)
    collection = collection.to_unit("radians")

    return collection


def solar_altitude(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar altitude angles.
    """

    if not sun_position:
        sun_position = sun_position_collection(epw)

    return HourlyContinuousCollection(
        Header(
            data_type=Angle(
                name="Solar Altitude",
            ),
            unit="degrees",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        [i.altitude for i in sun_position.values],
    )


def solar_altitude_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of solar altitude angles.
    """

    collection = solar_altitude(epw, sun_position)
    collection = collection.to_unit("radians")

    return collection


def radiation_tilt_orientation_matrix(
    epw: EPW, n_altitudes: int = 10, n_azimuths: int = 19
) -> pd.DataFrame:
    """Compute the annual cumulative radiation matrix per surface tilt and orientation, for a
        given EPW object.
    Args:
        epw (EPW):
            The EPW object for which this calculation is made.
        n_altitudes (int, optional):
            The number of altitudes between 0 and 90 to calculate. Default is 10.
        n_azimuths (int, optional):
            The number of azimuths between 0 and 360 to calculate. Default is 19.
    Returns:
        pd.DataFrame:
            A table of insolation values for each simulated azimuth and tilt combo.
    """
    wea = Wea.from_annual_values(
        epw.location,
        epw.direct_normal_radiation.values,
        epw.diffuse_horizontal_radiation.values,
        is_leap_year=epw.is_leap_year,
    )
    # I do a bit of a hack here, to calculate only the Eastern insolation - then mirror it about
    # the North-South axis to get the whole matrix
    altitudes = np.linspace(0, 90, n_altitudes)
    azimuths = np.linspace(0, 180, n_azimuths)
    combinations = np.array(list(itertools.product(altitudes, azimuths)))

    def f(alt_az):
        return copy.copy(wea).directional_irradiance(alt_az[0], alt_az[1])[0].total

    with ThreadPoolExecutor() as executor:
        results = np.array(list(executor.map(f, combinations[0:]))).reshape(
            len(altitudes), len(azimuths)
        )
    temp = pd.DataFrame(results, index=altitudes, columns=azimuths)
    new_cols = (360 - temp.columns)[::-1][1:]
    new_vals = temp.values[::-1, ::-1][
        ::-1, 1:
    ]  # some weird array transformation stuff here
    mirrored = pd.DataFrame(new_vals, columns=new_cols, index=temp.index)
    return pd.concat([temp, mirrored], axis=1)


def humidity_ratio(epw: EPW) -> HourlyContinuousCollection:
    """Calculate an annual hourly humidity ratio for a given EPW.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of humidity ratios.
    """
    return HourlyContinuousCollection.compute_function_aligned(
        humid_ratio_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        HumidityRatio(),
        "fraction",
    )


def from_dict(dictionary: Dict[str, Any]) -> EPW:
    """Convert a JSON compliant dictionary object into a ladybug EPW.

    Args:
        Dict[str, Any]:
            A sanitised dictionary.

    Returns:
        epw (EPW):
            An EPW object.
    """

    json_str = json.dumps(dictionary)

    # custom handling of non-standard JSON NaN/Inf values
    json_str = json_str.replace('"min": "-inf"', '"min": -Infinity')
    json_str = json_str.replace('"max": "inf"', '"max": Infinity')

    # custom handling of float-indexed values
    for k, _ in dictionary["monthly_ground_temps"].items():
        _new = k.replace("_", ".")[1:]
        json_str = json_str.replace(f'"{k}": {{', f'"{_new}": {{')

    return EPW.from_dict(json.loads(json_str))


def filename(epw: EPW, include_extension: bool = False) -> str:
    """Get the filename of the given EPW.

    Args:
        epw (EPW):
            An EPW object.
        include_extension (bool, optional):
            Set to True to include the file extension. Defaults to False.

    Returns:
        string:
            The name of the EPW file.
    """

    if include_extension:
        return Path(epw.file_path).name

    return Path(epw.file_path).stem


def equality(epw0: EPW, epw1: EPW, include_header: bool = False) -> bool:
    """Check for equality between two EPW objects, with regards to the data contained within.

    Args:
        epw0 (EPW):
            A ladybug EPW object.
        epw1 (EPW):
            A ladybug EPW object.
        include_header (bool, optional):
            Include the str representation of the EPW files header in the comparison.
            Defaults to False.

    Returns:
        bool:
            True if the two EPW objects are equal, False otherwise.
    """

    if not isinstance(epw0, EPW) or not isinstance(epw1, EPW):
        raise TypeError("Both inputs must be of type EPW.")

    if include_header:
        if epw0.header != epw1.header:
            return False

    # Check key metrics
    for var in [
        "dry_bulb_temperature",
        "relative_humidity",
        "dew_point_temperature",
        "wind_speed",
        "wind_direction",
        "global_horizontal_radiation",
        "direct_normal_radiation",
        "diffuse_horizontal_radiation",
        "atmospheric_station_pressure",
    ]:
        if getattr(epw0, var) != getattr(epw1, var):
            warnings.warn(f"{var}: {epw0} != {epw1}")
            return False

    return True


def equation_of_time(epw: EPW) -> HourlyContinuousCollection:
    """Calculate the equation of time for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of equation of times.
    """
    sunpath = Sunpath.from_location(epw.location)

    _, eot = list(
        zip(
            *[
                list(sunpath._calculate_solar_geometry(i))
                for i in epw.dry_bulb_temperature.datetimes
            ]
        )
    )

    return HourlyContinuousCollection(
        Header(
            data_type=Time(name="Equation of Time"),
            unit="min",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        eot,
    )


def epw_content_check(epw: EPW, fields: List[str] = None) -> bool:
    """Check an EPW object for whether it contains all valid fields

    Args:
        epw (EPW):
            An EPW object.
        fields (List[str], optional):
            The fields subset to check.

    Returns:
        bool:
            True if EPW File contains valid fields.
    """

    if fields is None:
        fields = [
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            # "Uncertainty Flags",
            "Dry Bulb Temperature",
            "Dew Point Temperature",
            "Relative Humidity",
            "Atmospheric Station Pressure",
            # "Extraterrestrial Horizontal Radiation",
            # "Extraterrestrial Direct Normal Radiation",
            "Horizontal Infrared Radiation Intensity",
            "Global Horizontal Radiation",
            "Direct Normal Radiation",
            "Diffuse Horizontal Radiation",
            # "Global Horizontal Illuminance",
            # "Direct Normal Illuminance",
            # "Diffuse Horizontal Illuminance",
            # "Zenith Luminance",
            "Wind Direction",
            "Wind Speed",
            "Total Sky Cover",
            "Opaque Sky Cover",
            # "Visibility",
            # "Ceiling Height",
            # "Present Weather Observation",
            # "Present Weather Codes",
            # "Precipitable Water",
            # "Aerosol Optical Depth",
            # "Snow Depth",
            # "Days Since Last Snowfall",
            # "Albedo",
            # "Liquid Precipitation Depth",
            # "Liquid Precipitation Quantity",
        ]
    valid = True
    epw_fields = EPWFields()
    for field_no in range(35):
        epw_field = epw_fields.field_by_number(field_no)
        if str(epw_field.name) not in fields:
            continue
        if all(i == epw_field.missing for i in epw._get_data_by_field(field_no)):
            warnings.warn(
                f"{epw} - {epw_field.name} contains only missing values.", stacklevel=2
            )
            valid = False
    return valid


def enthalpy(
    epw: EPW, hr: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate an annual hourly enthalpy for a given EPW.

    Args:
        epw (EPW):
            An EPW object.
        humidity_ratio (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of humidity ratios. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of enthalpies.
    """

    if not hr:
        hr = humidity_ratio(epw)

    return HourlyContinuousCollection.compute_function_aligned(
        enthalpy_from_db_hr,
        [
            epw.dry_bulb_temperature,
            hr,
        ],
        Enthalpy(),
        "kJ/kg",
    )


def clearness_index(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate the clearness index value for each hour of the year.

    Args:
        epw (EPW):
            An EPW object.
        sun_position (HourlyContinuousCollection, optional):
            A pre-calculated HourlyContinuousCollection of Sun objects. Defaults to None.

    Returns:
        HourlyContinuousCollection:
            An HourlyContinuousCollection of clearness indices.
    """

    if not sun_position:
        sun_position = sun_position_collection(epw)

    ci = []
    for i, j, k in list(
        zip(
            *[
                epw.global_horizontal_radiation,
                solar_altitude(epw, sun_position),
                epw.extraterrestrial_direct_normal_radiation,
            ]
        )
    ):
        try:
            ci.append(lb_ci(i, j, k))
        except ZeroDivisionError:
            ci.append(0)

    return HourlyContinuousCollection(
        header=Header(
            data_type=Fraction(name="Clearness Index"),
            unit="fraction",
            analysis_period=AnalysisPeriod(),
        ),
        values=ci,
    )


def seasonality_from_day_length(epw: EPW, annotate: bool = False) -> pd.Series:
    """Create a Series containing a category for each timestep of an EPW giving it's season based on day length (using sunrise/sunset).

    Args:
        epw (EPW):
            Input EPW.

    Returns:
        pd.Series:
            List of seasons per timestep.
    """

    # get datetimes to query sun
    idx = to_datetimes(AnalysisPeriod())
    df = pd.Series([0] * len(idx), index=idx).resample("1D").mean()

    sun_times = pd.DataFrame.from_dict(
        [
            Sunpath.from_location(epw.location).calculate_sunrise_sunset(i.month, i.day)
            for i in df.index
        ]
    )
    sun_times.index = df.index
    sun_times["day_length"] = sun_times.sunset - sun_times.sunrise

    if sun_times.sunset.isna().sum():
        warnings.warn(
            "This location is near the north/south pole and is subject to periods where sun neither rises or sets."
        )

    def _timedelta_to_str(timedelta: datetime.timedelta) -> str:
        s = timedelta.seconds
        hours, remainder = divmod(s, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}"

    shortest_day_length = _timedelta_to_str(sun_times.day_length.min())
    shortest_day = sun_times.day_length.idxmin()
    middlest_day_length = _timedelta_to_str(sun_times.day_length.mean())
    longest_day_length = _timedelta_to_str(sun_times.day_length.max())
    longest_day = sun_times.day_length.idxmax()

    months = pd.Timedelta(days=3 * 30)

    if (longest_day + months).year != longest_day.year:
        spring_equinox = shortest_day + months
        autumn_equinox = shortest_day - months

        autumn_right = autumn_equinox + (months / 2)
        autumn_left = autumn_equinox - (months / 2)
        spring_right = spring_equinox + (months / 2)
        spring_left = spring_equinox - (months / 2)

        spring_mask = (idx > spring_left) & (idx <= spring_right)
        summer_mask = (idx > spring_right) | (idx <= autumn_left)
        autumn_mask = (idx > autumn_left) & (idx <= autumn_right)
        winter_mask = (idx > autumn_right) & (idx <= spring_left)

    else:
        spring_equinox = longest_day - months
        autumn_equinox = longest_day + months

        autumn_right = autumn_equinox + (months / 2)
        autumn_left = autumn_equinox - (months / 2)
        spring_right = spring_equinox + (months / 2)
        spring_left = spring_equinox - (months / 2)

        spring_mask = (idx > spring_left) & (idx <= spring_right)
        summer_mask = (idx > spring_right) & (idx <= autumn_left)
        autumn_mask = (idx > autumn_left) & (idx <= autumn_right)
        winter_mask = (idx > autumn_right) | (idx <= spring_left)

    # construct datetime indexed series with categories
    categories = np.where(
        spring_mask,
        f"Spring ({middlest_day_length} average sun-up time, {spring_left:%b %d} to {spring_right: %b %d})"
        if annotate
        else "Spring",
        np.where(
            summer_mask,
            f"Summer ({longest_day_length} average sun-up time, {spring_right:%b %d} to {autumn_left: %b %d})"
            if annotate
            else "Summer",
            np.where(
                autumn_mask,
                f"Autumn ({middlest_day_length} average sun-up time, {autumn_left:%b %d} to {autumn_right: %b %d})"
                if annotate
                else "Autumn",
                np.where(
                    winter_mask,
                    f"Winter ({shortest_day_length} average sun-up time, {autumn_right:%b %d} to {spring_left: %b %d})"
                    if annotate
                    else "Winter",
                    "Undefined",
                ),
            ),
        ),
    )

    return pd.Series(categories, index=idx, name="season")


def seasonality_from_month(epw: EPW, annotate: bool = False) -> pd.Series:
    """Create a Series containing a category for each timestep of an EPW giving it's season.

    Args:
        epw (EPW):
            Input EPW.
        annotate (bool, optional):
            If True, then note months included in each season in the category labels.

    Returns:
        pd.Series:
            List of seasons per timestep.
    """

    idx = to_datetimes(AnalysisPeriod())
    if epw.location.latitude >= 0:
        # northern hemisphere
        spring_months = [3, 4, 5]
        spring_month_labels = [calendar.month_abbr[i] for i in spring_months]
        summer_months = [6, 7, 8]
        summer_month_labels = [calendar.month_abbr[i] for i in summer_months]
        autumn_months = [9, 10, 11]
        autumn_month_labels = [calendar.month_abbr[i] for i in autumn_months]
        winter_months = [12, 1, 2]
        winter_month_labels = [calendar.month_abbr[i] for i in winter_months]
    else:
        # southern hemisphere
        spring_months = [9, 10, 11]
        spring_month_labels = [calendar.month_abbr[i] for i in spring_months]
        summer_months = [12, 1, 2]
        summer_month_labels = [calendar.month_abbr[i] for i in summer_months]
        autumn_months = [3, 4, 5]
        autumn_month_labels = [calendar.month_abbr[i] for i in autumn_months]
        winter_months = [6, 7, 8]
        winter_month_labels = [calendar.month_abbr[i] for i in winter_months]

    categories = np.where(
        idx.month.isin(spring_months),
        f"Spring ({', '.join(spring_month_labels)})" if annotate else "Spring",
        np.where(
            idx.month.isin(summer_months),
            f"Summer ({', '.join(summer_month_labels)})" if annotate else "Summer",
            np.where(
                idx.month.isin(autumn_months),
                f"Autumn ({', '.join(autumn_month_labels)})" if annotate else "Autumn",
                np.where(
                    idx.month.isin(winter_months),
                    f"Winter ({', '.join(winter_month_labels)})"
                    if annotate
                    else "Winter",
                    "Undefined",
                ),
            ),
        ),
    )

    return pd.Series(categories, index=idx, name="season")


def seasonality_from_temperature(epw: EPW, annotate: bool = False) -> pd.Series:
    """Create a Series containing a category for each timestep of an EPW giving it's season.
    Args:
        epw (EPW):
            Input EPW.
    Returns:
        pd.Series:
            List of seasons per timestep.
    """
    dbt = to_series(epw.dry_bulb_temperature).rename("dbt")
    new_idx = pd.date_range(
        f"{dbt.index[0].year - 1}-01-01 00:00:00", freq="60T", periods=len(dbt) * 3
    )
    dbt_3year = pd.Series(
        index=new_idx, data=np.array([[i] * 3 for i in dbt.values]).T.flatten()
    )

    # check that weatherfile is "seasonal", by checking avg variance
    if dbt.std() <= 2.5:
        warnings.warn(
            "Input dataset has a low variance, indicating that seasonality may not be determined accurately from dry-bulb temperature."
        )

    # resample to weeks to get min and max week, and then min/max datetime within the middle of that week
    dbt_week_mean = dbt.resample("1W").mean()
    peak_summer = (dbt_week_mean.idxmax() + pd.Timedelta(days=3)).date()
    if peak_summer.year != dbt.index[0].year:
        peak_summer = datetime.datetime(
            dbt.index[0].year, peak_summer.month, peak_summer.day
        ).date()
    peak_winter = (dbt_week_mean.idxmin() + pd.Timedelta(days=3)).date()
    if peak_winter.year != dbt.index[0].year:
        peak_winter = datetime.datetime(
            dbt.index[0].year, peak_winter.month, peak_winter.day
        ).date()

    # get ranges of dates between peak days - regardless of whether they cross the year boundary
    if peak_summer > peak_winter:
        # print("summer later in year")
        summer_to_winter_3year_mask = (dbt_3year.index.date > peak_summer) & (
            dbt_3year.index.date < peak_winter + datetime.timedelta(days=365)
        )
        winter_to_summer_3year_mask = (dbt_3year.index.date <= peak_summer) & (
            dbt_3year.index.date > peak_winter
        )
    else:
        # print("summer earlier in year")
        winter_to_summer_3year_mask = (dbt_3year.index.date > peak_winter) & (
            dbt_3year.index.date < peak_summer + datetime.timedelta(days=365)
        )
        summer_to_winter_3year_mask = (dbt_3year.index.date <= peak_winter) & (
            dbt_3year.index.date > peak_summer
        )

    # create subset of 3-year data containing values where spring/autumn can be defined
    autumn_series = dbt_3year[summer_to_winter_3year_mask]
    spring_series = dbt_3year[winter_to_summer_3year_mask]

    # get the 25/75% values for the range
    autumn_lower_limit = autumn_series.quantile(0.25)
    spring_lower_limit = spring_series.quantile(0.25)
    autumn_upper_limit = autumn_series.quantile(0.75)
    spring_upper_limit = spring_series.quantile(0.75)

    # get the number of hours in each day in the mask where the majority of hours are within that band
    autumn_temp = (
        autumn_series.between(autumn_lower_limit, autumn_upper_limit, inclusive="left")
        .groupby(autumn_series.index.date)
        .sum()
        > 12
    )
    spring_temp = (
        spring_series.between(spring_lower_limit, spring_upper_limit, inclusive="left")
        .groupby(spring_series.index.date)
        .sum()
        > 12
    )

    # convert timestamps to epoch and get the weighted mean date for those dates where it's "spring"/"autumn"-y
    autumn_epoch = pd.to_datetime(autumn_temp.index).view("int64").astype("float64")
    autumn_weighted_mean_date = pd.to_datetime(
        (autumn_epoch * autumn_temp.astype(int)).sum() / autumn_temp.sum()
    ).date()
    peak_autumn = datetime.datetime(
        dbt.index.year[0],
        autumn_weighted_mean_date.month,
        autumn_weighted_mean_date.day,
    ).date()

    spring_epoch = pd.to_datetime(spring_temp.index).view("int64").astype("float64")
    spring_weighted_mean_date = pd.to_datetime(
        (spring_epoch * spring_temp.astype(int)).sum() / spring_temp.sum()
    ).date()
    peak_spring = datetime.datetime(
        dbt.index.year[0],
        spring_weighted_mean_date.month,
        spring_weighted_mean_date.day,
    ).date()

    # determine the midpoints between seasonal peaks to get the ranges of dates over which seasons occur
    if peak_spring > peak_winter:
        # print("spring happens after winter in the same year")
        spring_start = peak_spring + (peak_winter - peak_spring) / 2
    else:
        # print("spring happens after winter in a different year")
        _peak_winter = datetime.datetime(
            peak_winter.year - 1, peak_winter.month, peak_winter.day
        ).date()
        spring_start = peak_spring + (_peak_winter - peak_spring) / 2
        if spring_start.year != dbt.index[0].year:
            spring_start = datetime.datetime(
                dbt.index[0].year, spring_start.month, spring_start.day
            ).date()

    if peak_autumn > peak_summer:
        # print("autumn happens after summer in the same year")
        autumn_start = peak_autumn + (peak_summer - peak_autumn) / 2
    else:
        # print("autumn happens after summer in a different year")
        _peak_summer = datetime.datetime(
            peak_summer.year - 1, peak_summer.month, peak_summer.day
        ).date()
        autumn_start = peak_autumn + (_peak_summer - peak_autumn) / 2
        if autumn_start.year != dbt.index[0].year:
            autumn_start = datetime.datetime(
                dbt.index[0].year, autumn_start.month, autumn_start.day
            ).date()

    if peak_winter > peak_autumn:
        # print("winter happens after autumn in the same year")
        winter_start = peak_winter + (peak_autumn - peak_winter) / 2
    else:
        # print("winter happens after autumn in a different year")
        _peak_autumn = datetime.datetime(
            peak_autumn.year - 1, peak_autumn.month, peak_autumn.day
        ).date()
        winter_start = peak_winter + (_peak_autumn - peak_winter) / 2
        if winter_start.year != dbt.index[0].year:
            winter_start = datetime.datetime(
                dbt.index[0].year, winter_start.month, winter_start.day
            ).date()

    if peak_summer > peak_spring:
        # print("summer happens after spring in the same year")
        summer_start = peak_summer + (peak_spring - peak_summer) / 2
    else:
        # print("summer happens after spring in a different year")
        _peak_spring = datetime.datetime(
            peak_spring.year - 1, peak_spring.month, peak_spring.day
        ).date()
        summer_start = peak_summer + (_peak_spring - peak_summer) / 2
        if summer_start.year != dbt.index[0].year:
            summer_start = datetime.datetime(
                dbt.index[0].year, summer_start.month, summer_start.day
            ).date()

    temp = dbt.to_frame()
    s = pd.Series(index=dbt.index, data=np.nan)
    s.loc[spring_start] = "Spring"
    s.loc[summer_start] = "Summer"
    s.loc[autumn_start] = "Autumn"
    s.loc[winter_start] = "Winter"
    s.ffill(inplace=True)
    s.loc[temp.index == temp.index.min()] = s.iloc[-1]
    s.ffill(inplace=True)

    # get mean temps if annotated
    if annotate:
        s[
            s == "Spring"
        ] = f"Spring ({dbt[s == 'Spring'].mean():0.1f}°C average temperature, {spring_start:%b %d} to {summer_start:%b %d})"
        s[
            s == "Summer"
        ] = f"Summer ({dbt[s == 'Summer'].mean():0.1f}°C average temperature, {summer_start:%b %d} to {autumn_start:%b %d})"
        s[
            s == "Autumn"
        ] = f"Autumn ({dbt[s == 'Autumn'].mean():0.1f}°C average temperature, {autumn_start:%b %d} to {winter_start:%b %d})"
        s[
            s == "Winter"
        ] = f"Winter ({dbt[s == 'Winter'].mean():0.1f}°C average temperature, {winter_start:%b %d} to {spring_start:%b %d})"

    return s


def degree_time(
    epws: List[EPW],
    heat_base: float = 18,
    cool_base: float = 23,
    return_type: str = "days",
    names: List[str] = None,
):

    cooling_degree_time_v = np.vectorize(cooling_degree_time)
    heating_degree_time_v = np.vectorize(heating_degree_time)

    if heat_base > cool_base:
        warnings.warn("cool_base is lower than heat_base!")

    if not return_type.lower() in ["days", "hours"]:
        raise ValueError('return_type must be one of "days" or "hours".')

    if names is None:
        names = [Path(i.file_path).stem for i in epws]
    else:
        if len(names) != len(epws):
            raise ValueError("names must be the same length as epws.")
        names = [str(i) for i in names]

    df = pd.concat(
        [to_series(epw.dry_bulb_temperature) for epw in epws], axis=1, keys=names
    )

    cdh = pd.DataFrame(
        cooling_degree_time_v(df, cool_base), index=df.index, columns=df.columns
    )
    hdh = pd.DataFrame(
        heating_degree_time_v(df, heat_base), index=df.index, columns=df.columns
    )

    if return_type.lower() == "hours":
        return (
            pd.concat(
                [cdh, hdh],
                axis=1,
                keys=[
                    f"Cooling Degree Hours (>{cool_base})",
                    f"Heating Degree Hours (<{heat_base})",
                ],
            )
            .reorder_levels([1, 0], axis=1)
            .sort_index(axis=1)
        )

    cdd = cdh.resample("1D").sum() / 24
    hdd = hdh.resample("1D").sum() / 24

    return (
        pd.concat(
            [cdd, hdd],
            axis=1,
            keys=[
                f"Cooling Degree Days (>{cool_base})",
                f"Heating Degree Days (<{heat_base})",
            ],
        )
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
    )
