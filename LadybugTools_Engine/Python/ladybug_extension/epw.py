import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

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
from ladybug.sunpath import Sunpath
from ladybug.skymodel import clearness_index
from ladybug.wea import Wea

from ladybug_extension.datacollection import to_datetimes, to_hourly, to_series


def from_dataframe(dataframe: pd.DataFrame, location: Location = None) -> EPW:
    """Create an EPW object from a Pandas DataFrame with named columns."""

    # Check dataframe shapre for leapedness and length
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


def to_dataframe(
    epw: EPW,
) -> pd.DataFrame:
    """Create a Pandas DataFrame from an EPW object, including additional calculated properties."""

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
    sun_position = get_sun_position(epw)
    equation_of_time = get_equation_of_time(epw)
    solar_time_hour = get_solar_time_hour(epw, equation_of_time)
    solar_altitude = get_solar_altitude(epw, sun_position)
    solar_altitude_in_radians = get_solar_altitude_in_radians(epw, sun_position)
    solar_declination = get_solar_declination(epw)
    solar_time_datetime = get_solar_time_datetime(epw, solar_time_hour)
    solar_azimuth = get_solar_azimuth(epw, sun_position)
    solar_azimuth_in_radians = get_solar_azimuth_in_radians(epw, sun_position)
    apparent_solar_zenith = get_apparent_solar_zenith(epw, solar_altitude)
    clearness_index =get_clearness_index(epw, sun_position)

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
    ]:
        all_series.append(to_series(collection))

    # Compile all the data into a dataframe
    df = pd.concat(all_series, axis=1)  # .sort_index(axis=1)
    # df.columns = pd.MultiIndex.from_tuples(df.columns, names=('variable', 'unit', 'location'))

    return df


def get_sun_position(epw: EPW) -> HourlyContinuousCollection:
    """Calculate a set of Sun positions for each hour of the year."""
    sunpath = Sunpath.from_location(epw.location)

    suns = [sunpath.calculate_sun_from_hoy(i) for i in range(8760)]

    return HourlyContinuousCollection(
        Header(
            data_type=datatype.generic.GenericType(name="Sun Position", unit="Sun"),
            unit="Sun",
            analysis_period=AnalysisPeriod(),
            metadata=epw.dry_bulb_temperature.header.metadata,
        ),
        suns,
    )


def get_solar_declination(epw: EPW) -> HourlyContinuousCollection:
    """Calculate solar declination for each hour of the year."""
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


def get_equation_of_time(epw: EPW) -> HourlyContinuousCollection:
    """Calculate the equation of time for each hour of the year."""
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


def get_solar_time_hour(
    epw: EPW, equation_of_time: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (in hour-of-day) for each hour of the year."""

    if equation_of_time is None:
        equation_of_time = get_equation_of_time(epw)

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


def get_solar_time_datetime(
    epw: EPW, solar_time_hour: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate solar time (as datetime) for each hour of the year."""

    if solar_time_hour is None:
        solar_time_hour = get_solar_time_hour(epw)

    timestamp_str = [
        f"{int(i):02d}:{int(np.floor((i*60) % 60)):02d}:{(i*3600) % 60:0.8f}"
        for i in solar_time_hour
    ]
    date_str = to_datetimes(epw.dry_bulb_temperature).strftime("%Y-%m-%d")
    datetimes = pd.to_datetime(
        [f"{ds} {ts}" for ds, ts in list(zip(*[date_str, timestamp_str]))]
    )

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


def get_solar_azimuth(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle."""

    if not sun_position:
        sun_position = get_sun_position(epw)

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


def get_solar_azimuth_in_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar azimuth angle in radians."""

    if not sun_position:
        sun_position = get_sun_position(epw)

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


def get_solar_altitude(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle."""

    if not sun_position:
        sun_position = get_sun_position(epw)

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


def get_solar_altitude_in_radians(
    epw: EPW, sun_position: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly solar altitude angle in radians."""

    if not sun_position:
        sun_position = get_sun_position(epw)

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


def get_apparent_solar_zenith(
    epw: EPW, solar_altitude: HourlyContinuousCollection = None
) -> HourlyContinuousCollection:
    """Calculate annual hourly apparent solar zenith angles."""

    if not solar_altitude:
        solar_altitude = get_solar_altitude(epw)

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
    """Calculate an annual hourly wet bulb temperature for a given EPW."""
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
    """Calculate an annual hourly humidity ratio for a given EPW."""
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
    """Calculate an annual hourly enthalpy for a given EPW."""

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

def get_clearness_index(epw: EPW, sun_position: HourlyContinuousCollection = None) -> HourlyContinuousCollection:
    """Calculate the clearness index value for each hour of the year."""

    if not sun_position:
        sun_position = get_sun_position(epw)
    
    ci = []
    for i, j, k in list(zip(*[epw.global_horizontal_radiation, get_solar_altitude(epw, sun_position), epw.extraterrestrial_direct_normal_radiation])):
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

if __name__ == "__main__":
    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    print(to_dataframe(epw))