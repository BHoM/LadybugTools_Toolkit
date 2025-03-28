"""Helper methods used throughout the ladybugtools_toolkit."""
# pylint: disable=C0302
# pylint: disable=E0401
import calendar
import contextlib
import copy
import io
import json
import math
import re
import urllib.request
import warnings
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from caseconverter import snakecase
from honeybee.config import folders as hb_folders
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.epw import (EPW, AnalysisPeriod, HourlyContinuousCollection,
                         Location)
from ladybug.psychrometrics import wet_bulb_from_db_rh
from ladybug.skymodel import (calc_horizontal_infrared, calc_sky_temperature,
                              estimate_illuminance_from_irradiance,
                              get_extra_radiation, zhang_huang_solar,
                              zhang_huang_solar_split)
from ladybug.sunpath import Sunpath
from ladybug_geometry.geometry2d import Vector2D
from meteostat import Hourly, Point
from python_toolkit.bhom.analytics import bhom_analytics
from .bhom.logging import CONSOLE_LOGGER
from .ladybug_extension.dt import lb_datetime_from_datetime

from python_toolkit.helpers import (
    validate_timeseries,
    sanitise_string,
    convert_keys_to_snake_case,
    DecayMethod,
    proximity_decay,
    timedelta_tostring,
    decay_rate_smoother,
    cardinality,
    angle_from_cardinal,
    angle_from_north,
    angle_to_vector,
    remove_leap_days,
    safe_filename
)

# pylint: enable=E0401

@bhom_analytics()
def default_hour_analysis_periods() -> list[AnalysisPeriod]:
    """A set of generic Analysis Period objects, spanning times of day."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        aps = [
            AnalysisPeriod(st_hour=5, end_hour=12, timestep=1),
            AnalysisPeriod(st_hour=13, end_hour=17, timestep=1),
            AnalysisPeriod(st_hour=18, end_hour=21, timestep=1),
            AnalysisPeriod(st_hour=22, end_hour=4, timestep=1),
        ]

    return aps


@bhom_analytics()
def default_month_analysis_periods() -> list[AnalysisPeriod]:
    """A set of generic Analysis Period objects, spanning month of year."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        aps = [
            AnalysisPeriod(st_month=12, end_month=2),
            AnalysisPeriod(st_month=3, end_month=5),
            AnalysisPeriod(st_month=6, end_month=8),
            AnalysisPeriod(st_month=9, end_month=11),
        ]

    return aps


@bhom_analytics()
def default_combined_analysis_periods() -> list[AnalysisPeriod]:
    """A set of generic Analysis Period objects, spanning combinations of time of day and month of year."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        aps = []
        for ap_time in default_hour_analysis_periods():
            for ap_month in default_month_analysis_periods():
                aps.append(
                    AnalysisPeriod(
                        st_month=ap_month.st_month,
                        end_month=ap_month.end_month,
                        st_hour=ap_time.st_hour,
                        end_hour=ap_time.end_hour,
                        timestep=ap_time.timestep,
                    )
                )

    return aps


@bhom_analytics()
def default_analysis_periods() -> list[AnalysisPeriod]:
    """A set of generic Analysis Period objects, spanning all predefined
    combinations of time of day and month of year."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        aps = [
            AnalysisPeriod(),
        ]
        aps.extend(default_month_analysis_periods())
        aps.extend(default_hour_analysis_periods())
        aps.extend(default_combined_analysis_periods())

    return aps


@bhom_analytics()
def chunks(lst: list[Any], chunksize: int):
    """Partition an iterable into lists of length "chunksize".

    Args:
        lst (list[Any]): The list to be partitioned.
        chunksize (int): The size of each partition.

    Yields:
        list[Any]: A list of length "chunksize".
    """
    for i in range(0, len(lst), chunksize):
        yield lst[i : i + chunksize]


@bhom_analytics()
def scrape_weather(
    station: str,
    start_date: str = "1970-01-01",
    end_date: str = None,
    interpolate: bool = False,
    resample: bool = False,
) -> pd.DataFrame:
    """Scrape historic data from global airport weather stations using their ICAO codes
        (https://en.wikipedia.org/wiki/list_of_airports_by_IATA_and_ICAO_code)

    Args:
        station (str):
            Airport ICAO code.
        start_date (str, optional):
            Date from which records will be searched. Defaults to "1970-01-01".
        end_date (str, optional):
            Date until which records will be searched. Defaults to None.
        interpolate (bool, optional):
            Set to True to interpolate gaps smaller than 2-hours. Defaults to False.
        resample (bool, optional):
            Set to True to resample the data to 0 and 30 minutes past the hour. Defaults to False.

    Returns:
        pd.DataFrame:
            A Pandas DataFrame containing time-indexed weather data.
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Scrape data from source website (https://mesonet.agron.iastate.edu/request/download.phtml)
    uri = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station}&"
        f"year1={start_date.year}&"
        f"month1={start_date.month}&"
        f"day1={start_date.day}&"
        f"year2={end_date.year}&"
        f"month2={end_date.month}&"
        f"day2={end_date.day}&"
        "tz=Etc%2FUTC&"
        "format=onlycomma&"
        "latlon=yes&"
        "elev=yes&"
        "missing=null&"
        "trace=null&"
        "direct=no&"
        "data=tmpc&"
        "data=dwpc&"
        "data=relh&"
        "data=drct&"
        "data=sknt&"
        "data=alti&"
        "data=p01m&"
        "data=vsby&"
        "data=skyc1&"
        "data=skyc2&"
        "data=skyc3"
    )
    df = pd.read_csv(
        uri,
        header=0,
        index_col="valid",
        parse_dates=True,
        na_values=["M", "null"],
        low_memory=False,
    )

    # Post-process data into right units
    df["sknt"] = df.sknt / 1.94384  # convert knots to m/s
    # convert inches of mercury (Hg) to Pa
    df["alti"] = df.alti * 3386.38866667
    df["vsby"] = df["vsby"] * 1.60934  # convert miles to kilometres

    # Get sky clearness
    rplc = {
        "   ": 0,
        "CLR": 0,
        "NCD": 0,
        "NSC": 0,
        "SKC": 0,
        "///": 0,
        "FEW": 1.5,
        "SCT": 3.5,
        "BKN": 6,
        "OVC": 8,
        "VV ": 8,
        "VV": 8,
    }

    for i in ["skyc1", "skyc2", "skyc3"]:
        df[i] = df[i].fillna("NSC").replace(rplc) / 8 * 10
    df["opaque_sky_cover"] = df[["skyc1", "skyc2", "skyc3"]].mean(axis=1)
    df.drop(["skyc1", "skyc2", "skyc3"], axis=1, inplace=True)

    # Rename headers
    renamer = {
        "lon": "longitude",
        "lat": "latitude",
        "elevation": "elevation",
        "tmpc": "dry_bulb_temperature",
        "dwpc": "dew_point_temperature",
        "relh": "relative_humidity",
        "drct": "wind_direction",
        "sknt": "wind_speed",
        "alti": "atmospheric_station_pressure",
        "p01m": "liquid_precipitation_depth",
        "vsby": "visibility",
    }
    df.rename(columns=renamer, inplace=True)
    df.index.name = None

    # Calculate HIR and sky temperature
    df["horizontal_infrared_radiation_intensity"] = [
        calc_horizontal_infrared(
            row.opaque_sky_cover, row.dry_bulb_temperature, row.dew_point_temperature
        )
        for row in df.itertuples()
    ]
    df["sky_temperature"] = [
        calc_sky_temperature(
            row.horizontal_infrared_radiation_intensity, source_emissivity=1
        )
        for row in df.itertuples()
    ]

    # Calculate sun locations
    loc = Location(
        latitude=df.latitude.values[0],
        longitude=df.longitude.values[0],
        elevation=df.elevation.values[0],
        city=f"ICAO-{df.station.values[0]}",
    )
    altitude_in_radians = []
    azimuth_in_radians = []
    for i in df.index:
        sunpath = Sunpath.from_location(loc).calculate_sun_from_date_time(i)
        altitude_in_radians.append(sunpath.altitude_in_radians)
        azimuth_in_radians.append(sunpath.azimuth_in_radians)
    df["solar_altitude"] = altitude_in_radians
    df["solar_azimuth"] = azimuth_in_radians

    # Calculate irradiance and illuminance
    df["temp_offset_3"] = df.dry_bulb_temperature.shift(3)
    dir_norm, dif_horiz = zhang_huang_solar_split(
        df.solar_altitude * 180 / math.pi,
        df.index.day_of_year,
        df.opaque_sky_cover,
        df.relative_humidity,
        df.dry_bulb_temperature,
        df.temp_offset_3,
        df.wind_speed,
        df.atmospheric_station_pressure,
    )
    df["direct_normal_radiation"] = dir_norm
    df["diffuse_horizontal_radiation"] = dif_horiz
    df["global_horizontal_radiation"] = [
        zhang_huang_solar(
            row.solar_altitude * 180 / math.pi,
            row.opaque_sky_cover,
            row.relative_humidity,
            row.dry_bulb_temperature,
            row.temp_offset_3,
            row.wind_speed,
            irr_0=1355,
        )
        for row in df.itertuples()
    ]
    df["extraterrestrial_horizontal_radiation"] = [
        get_extra_radiation(i) for i in df.index.day_of_year
    ]
    df["extraterrestrial_horizontal_radiation"] = df[
        "extraterrestrial_horizontal_radiation"
    ].where(df.global_horizontal_radiation != 0, 0)
    df["direct_normal_radiation"].fillna(0, inplace=True)
    df["diffuse_horizontal_radiation"].fillna(0, inplace=True)
    df["global_horizontal_radiation"].fillna(0, inplace=True)

    vals = []
    for _, row in df.iterrows():
        vals.append(
            estimate_illuminance_from_irradiance(
                row.solar_altitude * 180 / math.pi,
                row.global_horizontal_radiation,
                row.direct_normal_radiation,
                row.diffuse_horizontal_radiation,
                row.dew_point_temperature,
            )
        )
    gh_ill, dn_ill, dh_ill, z_lum = list(zip(*vals))
    df["direct_normal_illuminance"] = dn_ill
    df["diffuse_horizontal_illuminance"] = dh_ill
    df["global_horizontal_illuminance"] = gh_ill
    df["zenith_luminance"] = z_lum
    df.drop(["temp_offset_3"], axis=1, inplace=True)

    if interpolate:
        df.interpolate(limit=4, inplace=True)

    if resample:
        df = df.resample("30min").mean()

    return df

@bhom_analytics()
def rolling_window(array: list[Any], window: int):
    """Throwaway function here to roll a window along a list.

    Args:
        array (list[Any]):
            A 1D list of some kind.
        window (int):
            The size of the window to apply to the list.

    Example:
        For an input list like [0, 1, 2, 3, 4, 5, 6, 7, 8],
        returns [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]

    Returns:
        list[list[Any]]:
            The resulting, "windowed" list.
    """

    if window > len(array):
        raise ValueError("Array length must be larger than window size.")

    a: np.ndarray = np.array(array)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)  # pylint: disable=unsubscriptable-object
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def epw_wind_vectors(epw: EPW, normalise: bool = False) -> list[Vector2D]:
    """Return a list of vectors from the EPW wind direction and speed.

    Args:
        epw (EPW):
            An EPW object.
        normalise (bool, optional):
            Normalise the vectors. Defaults to False.

    Returns:
        list[Vector2D]:
            A list of vectors.
    """

    wind_direction = np.array(epw.wind_direction)
    vectors = np.array(angle_to_vector(wind_direction))

    if not normalise:
        vectors *= np.array(epw.wind_speed)

    return [Vector2D(*i) for i in vectors.T]


class OpenMeteoVariable(Enum):
    """An enumeration of variables available from OpenMeteo, and their metadata for handling returned values."""

    TEMPERATURE_2M = auto()
    DEWPOINT_2M = auto()
    RELATIVEHUMIDITY_2M = auto()
    SURFACE_PRESSURE = auto()
    SHORTWAVE_RADIATION = auto()
    # DIRECT_RADIATION = auto()
    DIFFUSE_RADIATION = auto()
    WINDDIRECTION_10M = auto()
    WINDSPEED_10M = auto()
    CLOUDCOVER = auto()
    PRECIPITATION = auto()
    RAIN = auto()
    # SNOWFALL = auto()
    # CLOUDCOVER_LOW = auto()
    # CLOUDCOVER_MID = auto()
    # CLOUDCOVER_HIGH = auto()
    DIRECT_NORMAL_IRRADIANCE = auto()
    WINDSPEED_100M = auto()
    WINDDIRECTION_100M = auto()
    # WINDGUSTS_10M = auto()
    ET0_FAO_EVAPOTRANSPIRATION = auto()
    # VAPOR_PRESSURE_DEFICIT = auto()
    SOIL_TEMPERATURE_0_TO_7CM = auto()
    SOIL_TEMPERATURE_7_TO_28CM = auto()
    SOIL_TEMPERATURE_28_TO_100CM = auto()
    SOIL_TEMPERATURE_100_TO_255CM = auto()
    # SOIL_MOISTURE_0_TO_7CM = auto()
    # SOIL_MOISTURE_7_TO_28CM = auto()
    # SOIL_MOISTURE_28_TO_100CM = auto()
    # SOIL_MOISTURE_100_TO_255CM = auto()

    @staticmethod
    def __properties__() -> dict[str, dict[str, str | float]]:
        """A dictionary of the properties of each variable."""
        return {
            OpenMeteoVariable.TEMPERATURE_2M.value: {
                "openmeteo_name": "temperature_2m",
                "openmeteo_unit": "°C",
                "target_name": "Dry Bulb Temperature",
                "target_unit": "C",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.DEWPOINT_2M.value: {
                "openmeteo_name": "dewpoint_2m",
                "openmeteo_unit": "°C",
                "target_name": "Dew Point Temperature",
                "target_unit": "C",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.RELATIVEHUMIDITY_2M.value: {
                "openmeteo_name": "relativehumidity_2m",
                "openmeteo_unit": "%",
                "target_name": "Relative Humidity",
                "target_unit": "%",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SURFACE_PRESSURE.value: {
                "openmeteo_name": "surface_pressure",
                "openmeteo_unit": "hPa",
                "target_name": "Atmospheric Station Pressure",
                "target_unit": "Pa",
                "target_multiplier": 100,
            },
            OpenMeteoVariable.SHORTWAVE_RADIATION.value: {
                "openmeteo_name": "shortwave_radiation",
                "openmeteo_unit": "W/m²",
                "target_name": "Global Horizontal Radiation",
                "target_unit": "Wh/m2",
                "target_multiplier": 1,
            },
            # OpenMeteoVariable.DIRECT_RADIATION.value: {
            #     "openmeteo_name": "direct_radiation",
            #     "openmeteo_unit": "W/m²",
            #     "target_name": "Direct Horizontal Radiation",
            #     "target_unit": "Wh/m2",
            #     "target_multiplier": 1,
            # },
            OpenMeteoVariable.DIFFUSE_RADIATION.value: {
                "openmeteo_name": "diffuse_radiation",
                "openmeteo_unit": "W/m²",
                "target_name": "Diffuse Horizontal Radiation",
                "target_unit": "Wh/m2",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.WINDDIRECTION_10M.value: {
                "openmeteo_name": "winddirection_10m",
                "openmeteo_unit": "°",
                "target_name": "Wind Direction",
                "target_unit": "degrees",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.WINDSPEED_10M.value: {
                "openmeteo_name": "windspeed_10m",
                "openmeteo_unit": "km/h",
                "target_name": "Wind Speed",
                "target_unit": "m/s",
                "target_multiplier": 1 / 3.6,
            },
            OpenMeteoVariable.CLOUDCOVER.value: {
                "openmeteo_name": "cloudcover",
                "openmeteo_unit": "%",
                "target_name": "Opaque Sky Cover",
                "target_unit": "tenths",
                "target_multiplier": 0.1,
            },
            OpenMeteoVariable.PRECIPITATION.value: {
                "openmeteo_name": "precipitation",
                "openmeteo_unit": "mm",
                "target_name": "Precipitable Water",
                "target_unit": "mm",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.RAIN.value: {
                "openmeteo_name": "rain",
                "openmeteo_unit": "mm",
                "target_name": "Liquid Precipitation Depth",
                "target_unit": "mm",
                "target_multiplier": 1,
            },
            # OpenMeteoVariable.SNOWFALL.value: {
            #     "openmeteo_name": "snowfall",
            #     "openmeteo_unit": "cm",
            #     "target_name": "Snow depth",
            #     "target_unit": "cm",
            #     "target_multiplier": 1,
            # },
            # OpenMeteoVariable.CLOUDCOVER_LOW.value: {
            #     "openmeteo_name": "cloudcover_low",
            #     "openmeteo_unit": "%",
            #     "target_name": "Cloud Cover @<2km",
            #     "target_unit": "tenths",
            #     "target_multiplier": 0.1,
            # },
            # OpenMeteoVariable.CLOUDCOVER_MID.value: {
            #     "openmeteo_name": "cloudcover_mid",
            #     "openmeteo_unit": "%",
            #     "target_name": "Cloud Cover @2-6km",
            #     "target_unit": "tenths",
            #     "target_multiplier": 0.1,
            # },
            # OpenMeteoVariable.CLOUDCOVER_HIGH.value: {
            #     "openmeteo_name": "cloudcover_high",
            #     "openmeteo_unit": "%",
            #     "target_name": "Cloud Cover @>6km",
            #     "target_unit": "tenths",
            #     "target_multiplier": 0.1,
            # },
            OpenMeteoVariable.DIRECT_NORMAL_IRRADIANCE.value: {
                "openmeteo_name": "direct_normal_irradiance",
                "openmeteo_unit": "W/m²",
                "target_name": "Direct Normal Radiation",
                "target_unit": "Wh/m2",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.WINDSPEED_100M.value: {
                "openmeteo_name": "windspeed_100m",
                "openmeteo_unit": "km/h",
                "target_name": "Wind Speed @100m",
                "target_unit": "m/s",
                "target_multiplier": 1 / 3.6,
            },
            OpenMeteoVariable.WINDDIRECTION_100M.value: {
                "openmeteo_name": "winddirection_100m",
                "openmeteo_unit": "°",
                "target_name": "Wind Direction @100m",
                "target_unit": "degrees",
                "target_multiplier": 1,
            },
            # OpenMeteoVariable.WINDGUSTS_10M.value: {
            #     "openmeteo_name": "windgusts_10m",
            #     "openmeteo_unit": "km/h",
            #     "target_name": "Wind Gusts @10m",
            #     "target_unit": "m/s",
            #     "target_multiplier": 1 / 3.6,
            # },
            OpenMeteoVariable.ET0_FAO_EVAPOTRANSPIRATION.value: {
                "openmeteo_name": "et0_fao_evapotranspiration",
                "openmeteo_unit": "mm",
                "target_name": "Evapotranspiration",
                "target_unit": "mm/inch",
                "target_multiplier": 1,
            },
            # OpenMeteoVariable.VAPOR_PRESSURE_DEFICIT.value: {
            #     "openmeteo_name": "vapor_pressure_deficit",
            #     "openmeteo_unit": "kPa",
            #     "target_name": "Vapor Pressure Deficit",
            #     "target_unit": "Pa",
            #     "target_multiplier": 0.001,
            # },
            OpenMeteoVariable.SOIL_TEMPERATURE_0_TO_7CM.value: {
                "openmeteo_name": "soil_temperature_0_to_7cm",
                "openmeteo_unit": "°C",
                "target_name": "Soil Temperature @0-7cm",
                "target_unit": "C",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SOIL_TEMPERATURE_7_TO_28CM.value: {
                "openmeteo_name": "soil_temperature_7_to_28cm",
                "openmeteo_unit": "°C",
                "target_name": "Soil Temperature @7-28cm",
                "target_unit": "C",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SOIL_TEMPERATURE_28_TO_100CM.value: {
                "openmeteo_name": "soil_temperature_28_to_100cm",
                "openmeteo_unit": "°C",
                "target_name": "Soil Temperature @28-100cm",
                "target_unit": "C",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SOIL_TEMPERATURE_100_TO_255CM.value: {
                "openmeteo_name": "soil_temperature_100_to_255cm",
                "openmeteo_unit": "°C",
                "target_name": "Soil Temperature @100-255cm",
                "target_unit": "C",
                "target_multiplier": 1,
            },
            # OpenMeteoVariable.SOIL_MOISTURE_0_TO_7CM.value: {
            #     "openmeteo_name": "soil_moisture_0_to_7cm",
            #     "openmeteo_unit": "m³/m³",
            #     "target_name": "Soil Moisture @0-7cm",
            #     "target_unit": "fraction",
            #     "target_multiplier": 1,
            # },
            # OpenMeteoVariable.SOIL_MOISTURE_7_TO_28CM.value: {
            #     "openmeteo_name": "soil_moisture_7_to_28cm",
            #     "openmeteo_unit": "m³/m³",
            #     "target_name": "Soil Moisture @7-28cm",
            #     "target_unit": "fraction",
            #     "target_multiplier": 1,
            # },
            # OpenMeteoVariable.SOIL_MOISTURE_28_TO_100CM.value: {
            #     "openmeteo_name": "soil_moisture_28_to_100cm",
            #     "openmeteo_unit": "m³/m³",
            #     "target_name": "Soil Moisture @28-100cm",
            #     "target_unit": "fraction",
            #     "target_multiplier": 1,
            # },
            # OpenMeteoVariable.SOIL_MOISTURE_100_TO_255CM.value: {
            #     "openmeteo_name": "soil_moisture_100_to_255cm",
            #     "openmeteo_unit": "m³/m³",
            #     "target_name": "Soil Moisture @100-255cm",
            #     "target_unit": "fraction",
            #     "target_multiplier": 1,
            # },
        }

    @classmethod
    def from_string(cls, name: str) -> "OpenMeteoVariable":
        """."""
        d = {
            "temperature_2m": cls.TEMPERATURE_2M,
            "dewpoint_2m": cls.DEWPOINT_2M,
            "relativehumidity_2m": cls.RELATIVEHUMIDITY_2M,
            "surface_pressure": cls.SURFACE_PRESSURE,
            "shortwave_radiation": cls.SHORTWAVE_RADIATION,
            # "direct_radiation": cls.DIRECT_RADIATION,
            "diffuse_radiation": cls.DIFFUSE_RADIATION,
            "winddirection_10m": cls.WINDDIRECTION_10M,
            "windspeed_10m": cls.WINDSPEED_10M,
            "cloudcover": cls.CLOUDCOVER,
            "precipitation": cls.PRECIPITATION,
            "rain": cls.RAIN,
            # "snowfall": cls.SNOWFALL,
            # "cloudcover_low": cls.CLOUDCOVER_LOW,
            # "cloudcover_mid": cls.CLOUDCOVER_MID,
            # "cloudcover_high": cls.CLOUDCOVER_HIGH,
            "direct_normal_irradiance": cls.DIRECT_NORMAL_IRRADIANCE,
            "windspeed_100m": cls.WINDSPEED_100M,
            "winddirection_100m": cls.WINDDIRECTION_100M,
            # "windgusts_10m": cls.WINDGUSTS_10M,
            "et0_fao_evapotranspiration": cls.ET0_FAO_EVAPOTRANSPIRATION,
            # "vapor_pressure_deficit": cls.VAPOR_PRESSURE_DEFICIT,
            "soil_temperature_0_to_7cm": cls.SOIL_TEMPERATURE_0_TO_7CM,
            "soil_temperature_7_to_28cm": cls.SOIL_TEMPERATURE_7_TO_28CM,
            "soil_temperature_28_to_100cm": cls.SOIL_TEMPERATURE_28_TO_100CM,
            "soil_temperature_100_to_255cm": cls.SOIL_TEMPERATURE_100_TO_255CM,
            # "soil_moisture_0_to_7cm": cls.SOIL_MOISTURE_0_TO_7CM,
            # "soil_moisture_7_to_28cm": cls.SOIL_MOISTURE_7_TO_28CM,
            # "soil_moisture_28_to_100cm": cls.SOIL_MOISTURE_28_TO_100CM,
            # "soil_moisture_100_to_255cm": cls.SOIL_MOISTURE_100_TO_255CM,
        }
        try:
            return d[name]
        except KeyError as e:
            raise KeyError(e, f"{name} is not a known variable name.") from e

    @property
    def openmeteo_name(self) -> str:
        """The name of the variable as provided by OpenMeteo."""
        return self.__properties__()[self.value]["openmeteo_name"]

    @property
    def openmeteo_unit(self) -> str:
        """The unit of the variable as provided by OpenMeteo."""
        return self.__properties__()[self.value]["openmeteo_unit"]

    @property
    def openmeteo_table_name(self) -> str:
        """The name of the column header when placed into dataframe."""
        return f"{self.openmeteo_name} ({self.openmeteo_unit})"

    @property
    def target_table_name(self) -> str:
        """The name of the target column header when placed into dataframe."""
        return f"{self.target_name} ({self.target_unit})"

    @property
    def target_name(self) -> str:
        """The target name to convert to."""
        return self.__properties__()[self.value]["target_name"]

    @property
    def target_unit(self) -> str:
        """The target unit to convert to."""
        return self.__properties__()[self.value]["target_unit"]

    @property
    def target_multiplier(self) -> float:
        """The multiplier to convert from OpenMeteo units to target units."""
        return self.__properties__()[self.value]["target_multiplier"]

    def convert(self, value: float) -> float:
        """Convert the value from OpenMeteo units into target units."""
        return value * self.target_multiplier


@bhom_analytics()
def scrape_openmeteo(
    latitude: float,
    longitude: float,
    start_date: datetime | str,
    end_date: datetime | str,
    variables: tuple[OpenMeteoVariable] = None,
    convert_units: bool = False,
    remove_leapyears: bool = False,
) -> pd.DataFrame:
    """Obtain historic hourly data from Open-Meteo.
    https://open-meteo.com/en/docs/historical-weather-api

    Args:
        latitude (float):
            The latitude of the target site, in degrees.
        longitude (float):
            The longitude of the target site, in degrees.
        start_date (datetime | str):
            The start-date from which records will be obtained.
        end_date (datetime | str):
            The end-date beyond which records will be ignored.
        variables (tuple[OpenMeteoVariable]):
            A list of variables to query. If None, then all variables will be queried.
        convert_units (bool, optional):
            Convert units output into more common units, and rename headers accordingly.
        remove_leapyears (bool, optional):
            Whether or not to remove occurences of February 29th from the scraped data.

    Note:
        This method saves the data to a local cache, and will return the cached data if it is less
        than 100 days old. This is to avoid unnecessary API calls.

    Returns:
        pd.DataFrame:
            A DataFrame containing scraped data.
    """

    # sense checking, error handling and conversions
    if latitude < -90 or latitude > 90:
        raise ValueError("The latitude must be between -90 and 90 degrees.")
    if longitude < -180 or longitude > 180:
        raise ValueError("The longitude must be between -180 and 180 degrees.")

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if not isinstance(start_date, datetime):
        raise ValueError("The start_date must be a datetime object or string.")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if not isinstance(end_date, datetime):
        raise ValueError("The end_date must be a datetime object or string.")

    if start_date > end_date:
        raise ValueError("The start_date must be before the end_date.")

    if variables is None:
        variables = tuple(OpenMeteoVariable)
    # else:
        # TODO fix error that happens here
    #     if not all(isinstance(val, OpenMeteoVariable) for val in variables):
    #         raise ValueError(
    #             "All values in the variables tuple must be of type OpenMeteoVariable."
    #         )

    _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_openmeteo"
    _dir.mkdir(exist_ok=True, parents=True)

    # check _dir for existence of scraped data matching this query
    missing_variables = []
    available_data = []
    for var in variables:
        # TODO - add check in here for whether data exists as subset of longer time period within cache
        sp = (
            _dir
            / f"{latitude}_{longitude}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{var.name}.csv"
        )
        if sp.exists() and (
            (datetime.now() - datetime.fromtimestamp(sp.stat().st_mtime)).days <= 100
        ):
            CONSOLE_LOGGER.info(f"Reloading cached data for {var.name}")
            available_data.append(pd.read_csv(sp, index_col=0, parse_dates=True))
        else:
            CONSOLE_LOGGER.info(f"Querying data for {var.name}")
            missing_variables.append(var)

    if len(missing_variables) != 0:
        # construct query string
        var_strings = ",".join([var.openmeteo_name for var in missing_variables])
        query_string = (
            "https://archive-api.open-meteo.com/v1/era5?latitude="
            f"{latitude}&longitude={longitude}&start_date={start_date:%Y-%m-%d}&"
            f"end_date={end_date:%Y-%m-%d}&hourly={var_strings}"
        )

        # request data
        with urllib.request.urlopen(query_string) as url:
            data = json.load(url)

        # construct dataframe
        df = pd.DataFrame(data["hourly"])
        df.set_index("time", inplace=True, drop=True)
        df.index.name = None
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors="coerce")

        # write to cache
        for col in df.columns:
            sp = (_dir / f"{latitude}_{longitude}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{col}.csv")
            df[col].to_csv(sp)
            available_data.append(df[col])

    # combine data
    df = pd.concat(available_data, axis=1)

    if remove_leapyears:
        df = df[~((df.index.month == 2) & (df.index.day == 29))]

    if not convert_units:
        return df

    new_df = pd.DataFrame()
    for i in OpenMeteoVariable:
        try:
            # print(i.target_table_name, i.openmeteo_table_name, i.target_multiplier)
            new_df[i.target_table_name] = df[i.openmeteo_name] * i.target_multiplier
        except Exception as e:
            pass

    return new_df


def scrape_meteostat(
    latitude: float,
    longitude: float,
    start_date: datetime | str,
    end_date: datetime | str,
    altitude: float = None,
    convert_units: bool = False,
) -> pd.DataFrame:
    """Obtain historic hourly data from Meteostat.

    Args:
        latitude (float):
            The latitude of the target site, in degrees.
        longitude (float):
            The longitude of the target site, in degrees.
        start_date (datetime | str):
            The start-date from which records will be obtained.
        end_date (datetime | str):
            The end-date beyond which records will be ignored.
        altitude (float, optional):
            The altitude of the target site, in metres. Defaults to None.
        convert_units (bool, optional):
            Convert units output into more common units, and rename headers accordingly.

    Returns:
        pd.DataFrame:
            A DataFrame containing scraped data.
    """

    # TODO - implement caching in here, similar to how it's done for the OpenMeteo method

    if latitude < -90 or latitude > 90:
        raise ValueError("The latitude must be between -90 and 90 degrees.")
    if longitude < -180 or longitude > 180:
        raise ValueError("The longitude must be between -180 and 180 degrees.")

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if not isinstance(start_date, datetime):
        raise ValueError("The start_date must be a datetime object or string.")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if not isinstance(end_date, datetime):
        raise ValueError("The end_date must be a datetime object or string.")

    location = Point(latitude, longitude, altitude)
    data = Hourly(location, start_date, end_date).fetch()

    if len(data) == 0:
        raise ValueError("No data was returned from Meteostat.")

    if convert_units:
        converter = {
            "temp": [1, "Dry Bulb Temperature (C)"],
            "dwpt": [1, "Dew Point Temperature (C)"],
            "rhum": [1, "Relative Humidity (%)"],
            "prcp": [1, "Liquid Precipitation Depth (mm)"],
            "snow": [0.1, "Snow Depth (cm)"],
            "wdir": [1, "Wind Direction (degrees)"],
            "wspd": [1 / 3.6, "Wind Speed (m/s)"],
            "wpgt": [1 / 3.6, "Wind Gust (m/s)"],
            "pres": [100, "Atmospheric Station Pressure (Pa)"],
            "tsun": [1, "One Hour Sunshine (minutes)"],
            "coco": [1, "Present Weather (text)"],
        }
        weather_codes = {
            1: "Clear",
            2: "Fair",
            3: "Cloudy",
            4: "Overcast",
            5: "Fog",
            6: "Freezing Fog",
            7: "Light Rain",
            8: "Rain",
            9: "Heavy Rain",
            10: "Freezing Rain",
            11: "Heavy Freezing Rain",
            12: "Sleet",
            13: "Heavy Sleet",
            14: "Light Snowfall",
            15: "Snowfall",
            16: "Heavy Snowfall",
            17: "Rain Shower",
            18: "Heavy Rain Shower",
            19: "Sleet Shower",
            20: "Heavy Sleet Shower",
            21: "Snow Shower",
            22: "Heavy Snow Shower",
            23: "Lightning",
            24: "Hail",
            25: "Thunderstorm",
            26: "Heavy Thunderstorm",
            27: "Storm",
        }
        temp = []
        for col_name, col_values in data.items():
            if col_name == "coco":
                temp.append(
                    pd.Series(
                        col_values.map(weather_codes), name=converter[col_name][1]
                    )
                )
            else:
                temp.append(
                    (col_values * converter[col_name][0]).rename(converter[col_name][1])
                )
        return pd.concat(temp, axis=1)

    return data


@bhom_analytics()
def get_soil_temperatures(
    latitude: float,
    longitude: float,
    start_date: datetime | str,
    end_date: datetime | str,
    include_dbt: bool = False,
) -> pd.DataFrame:
    """Query Open-Meteo for soil temperature data.

    Args:
        latitude (float):
            The latitude of the target site, in degrees.
        longitude (float):
            The longitude of the target site, in degrees.
        start_date (datetime | str):
            The start-date from which records will be obtained.
        end_date (datetime | str):
            The end-date beyond which records will be ignored.
        include_dbt (bool, optional):
            Include dry bulb temperature in the output. Defaults to False.

    Returns:
        pd.DataFrame:
            A DataFrame containing scraped data.
    """

    variables = [
        OpenMeteoVariable.SOIL_TEMPERATURE_0_TO_7CM,
        OpenMeteoVariable.SOIL_TEMPERATURE_7_TO_28CM,
        OpenMeteoVariable.SOIL_TEMPERATURE_28_TO_100CM,
        OpenMeteoVariable.SOIL_TEMPERATURE_100_TO_255CM,
    ]
    if include_dbt:
        variables.append(OpenMeteoVariable.TEMPERATURE_2M)

    return scrape_openmeteo(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        convert_units=True,
    )


def circular_weighted_mean(angles: list[float], weights: list[float] = None):
    """Get the average angle from a set of weighted angles.

    Args:
        angles (list[float]):
            A collection of equally weighted wind directions, in degrees from North (0).
        weights (list[float]):
            A collection of weights, which must sum to 1. Defaults to None which will equally weight all angles.

    Returns:
        float:
            An average wind direction.
    """
    angles = np.array(angles)
    angles = np.where(angles == 360, 0, angles)

    if weights is None:
        weights = np.ones_like(angles) / len(angles)
    weights = np.array(weights)

    if angles.shape != weights.shape:
        raise ValueError("weights must be the same size as angles.")

    if np.any(angles < 0) or np.any(angles > 360):
        raise ValueError("Input angles exist outside of expected range (0-360).")

    # checks for opposing or equally spaced angles, with equal weighting
    if len(set(weights)) == 1:
        _sorted = np.sort(angles)
        if len(set(angles)) == 2:
            a, b = np.meshgrid(_sorted, _sorted)
            if np.any(a - b == 180):
                warnings.warn(
                    "Input angles are opposing, meaning determining the mean is impossible. An attempt will be made to determine the mean, but this will be perpendicular to the opposing angles and not accurate."
                )
        if any(np.diff(_sorted) == 360 / len(angles)):
            warnings.warn(
                "Input angles are equally spaced, meaning determining the mean is impossible. An attempt will be made to determine the mean, but this will not be accurate."
            )

    weights = np.array(weights) / sum(weights)

    x = y = 0.0
    for angle, weight in zip(angles, weights):
        x += np.cos(np.radians(angle)) * weight
        y += np.sin(np.radians(angle)) * weight

    mean = np.degrees(np.arctan2(y, x))

    if mean < 0:
        mean = 360 + mean

    if mean in (360.0, -0.0):
        mean = 0.0

    return np.round(mean, 5)


def wind_speed_at_height(
    reference_value: float,
    reference_height: float,
    target_height: float,
    **kwargs,
) -> float:
    """Calculate the wind speed at a given height from the 10m default height
        as stated in an EPW file.

    Args:
        reference_value (float):
            The speed to be translated.
        reference_height (float):
            The original height of the wind speed being translated.
        target_height (float):
            The target height of the wind speed being translated.

        **kwargs:
            Additional keyword arguments to pass to the translation method. These include:
            terrain_roughness_length (float):
                A value describing how rough the ground is. Default is
                0.03 for Open flat terrain; grass, few isolated obstacles.
            log_function (bool, optional):
                Set to True to used the log transformation method, or
                False for the exponent method. Defaults to True.

    Notes:
        Terrain roughness lengths can be found in the following table:

        +---------------------------------------------------+-----------+
        | Terrain description                               |  z0 (m)   |
        +===================================================+===========+
        | Open sea, Fetch at least 5 km                     |    0.0002 |
        +---------------------------------------------------+-----------+
        | Mud flats, snow; no vegetation, no obstacles      |    0.005  |
        +---------------------------------------------------+-----------+
        | Open flat terrain; grass, few isolated obstacle   |    0.03   |
        +---------------------------------------------------+-----------+
        | Low crops; occasional large obstacles, x/H > 20   |    0.10   |
        +---------------------------------------------------+-----------+
        | High crops; scattered obstacles, 15 < x/H < 20    |    0.25   |
        +---------------------------------------------------+-----------+
        | Parkland, bushes; numerous obstacles, x/H ≈ 10    |    0.5    |
        +---------------------------------------------------+-----------+
        | Regular large obstacle coverage (suburb, forest)  |    1.0    |
        +---------------------------------------------------+-----------+
        | City centre with high- and low-rise buildings     |    ≥ 2    |
        +---------------------------------------------------+-----------+


    Returns:
        float:
            The translated wind speed at the target height.
    """

    terrain_roughness_length = kwargs.get("terrain_roughness_length", 0.03)
    log_function = kwargs.get("log_function", True)
    kwargs = {}  # reset kwargs to remove invalid arguments

    if log_function:
        return reference_value * (
            np.log(target_height / terrain_roughness_length)
            / np.log(reference_height / terrain_roughness_length)
        )
    wind_shear_exponent = 1 / 7
    return reference_value * (
        np.power((target_height / reference_height), wind_shear_exponent)
    )


def temperature_at_height(
    reference_value: float,
    reference_height: float,
    target_height: float,
    **kwargs,
) -> float:
    # pylint: disable=C0301
    """Estimate the dry-bulb temperature at a given height from a referenced
        dry-bulb temperature at another height.

    Args:
        reference_value (float):
            The temperature to translate.
        reference_height (float):
            The height of the reference temperature.
        target_height (float):
            The height to translate the reference temperature towards.
        **kwargs:
            Additional keyword arguments to pass to the translation method. These include:
            reduction_per_km_altitude_gain (float, optional):
                The lapse rate of the atmosphere. Defaults to 0.0065 based
                on https://scied.ucar.edu/learning-zone/atmosphere/change-atmosphere-altitude#:~:text=Near%20the%20Earth's%20surface%2C%20air,standard%20(average)%20lapse%20rate
            lapse_rate (float, optional):
                The degrees C reduction for every 1 altitude gain. Default is 0.0065C for clear
                conditions (or 6.5C per 1km). This would be nearer 0.0098C/m if cloudy/moist air conditions.

    Returns:
        float:
            A translated air temperature.
    """
    # pylint: enable=C0301

    if (target_height > 8000) or (reference_height > 8000):
        warnings.warn(
            "The heights input into this calculation exist partially above "
            "the egde of the troposphere. This method is only valid below 8000m."
        )

    lapse_rate = kwargs.get("lapse_rate", 0.0065)
    kwargs = {}  # reset kwargs to remove invalid arguments

    height_difference = target_height - reference_height

    return reference_value - (height_difference * lapse_rate)


def radiation_at_height(
    reference_value: float,
    target_height: float,
    reference_height: float,
    **kwargs,
) -> float:
    """Calculate the radiation at a given height, given a reference
    radiation and height.

    References:
        Armel Oumbe, Lucien Wald. A parameterisation of vertical profile of
        solar irradiance for correcting solar fluxes for changes in terrain
        elevation. Earth Observation and Water Cycle Science Conference, Nov
        2009, Frascati, Italy. pp.S05.

    Args:
        reference_value (float):
            The radiation at the reference height.
        target_height (float):
            The height at which the radiation is required, in m.
        reference_height (float, optional):
            The height at which the reference radiation was measured.
        **kwargs:
            Additional keyword arguments to pass to the translation method. These include:
            lapse_rate (float, optional):
                The lapse rate of the atmosphere. Defaults to 0.08.

    Returns:
        float:
            The radiation at the given height.
    """
    lapse_rate = kwargs.get("lapse_rate", 0.08)
    kwargs = {}  # reset kwargs to remove invalid arguments

    lapse_rate_per_m = lapse_rate * reference_value / 1000
    increase = lapse_rate_per_m * (target_height - reference_height)
    return reference_value + increase


@bhom_analytics()
def air_pressure_at_height(
    reference_value: float,
    target_height: float,
    reference_height: float,
) -> float:
    """Calculate the air pressure at a given height, given a reference pressure and height.

    Args:
        reference_value (float):
            The pressure at the reference height, in Pa.
        target_height (float):
            The height at which the pressure is required, in m.
        reference_height (float, optional):
            The height at which the reference pressure was measured. Defaults to 10m.

    Returns:
        float:
            The pressure at the given height.
    """
    return (
        reference_value
        * (1 - 0.0065 * (target_height - reference_height) / 288.15) ** 5.255
    )


@bhom_analytics()
def target_wind_speed_collection(
    epw: EPW, target_average_wind_speed: float, target_height: float
) -> HourlyContinuousCollection:
    """Create an annual hourly collection of wind-speeds whose average equals the target value,
        translated to 10m height, using the source EPW to provide a wind-speed profile.

    Args:
        epw (EPW):
            The source EPW from which the wind speed profile is used to distribute wind speeds.
        target_average_wind_speed (float):
            The value to be translated to 10m and set as the average for the target wind-speed
            collection.
        target_height (float):
            The height at which the wind speed is translated to (this will assume the original wind
            speed is at 10m per EPW conventions.

    Returns:
        HourlyContinuousCollection:
            A ladybug annual hourly data wind speed collection.
    """

    # Translate target wind speed at ground level to wind speed at 10m, assuming an open terrain per airport conditions
    target_average_wind_speed_at_10m = wind_speed_at_height(
        reference_value=target_average_wind_speed,
        reference_height=target_height,
        target_height=10,
        terrain_roughness_length=0.03,
    )

    # Adjust hourly values in wind_speed to give a new overall average equal to that of the target wind-speed
    adjustment_factor = target_average_wind_speed_at_10m / epw.wind_speed.average

    return epw.wind_speed * adjustment_factor


@bhom_analytics()
def dry_bulb_temperature_at_height(
    epw: EPW, target_height: float
) -> HourlyContinuousCollection:
    """Translate DBT values from an EPW into

    Args:
        epw (EPW): A Ladybug EPW object.
        target_height (float): The height to translate the reference temperature towards.

    Returns:
        HourlyContinuousCollection: A resulting dry-bulb temperature collection.
    """
    dbt_collection = copy.copy(epw.dry_bulb_temperature)
    dbt_collection.values = [
        temperature_at_height(i, 10, target_height)
        for i in epw.dry_bulb_temperature.values
    ]
    return dbt_collection

def evaporative_cooling_effect(
    dry_bulb_temperature: float,
    relative_humidity: float,
    evaporative_cooling_effectiveness: float,
    atmospheric_pressure: float = None,
) -> list[float]:
    """
    For the inputs, calculate the effective DBT and RH values for the evaporative cooling
    effectiveness given.

    Args:
        dry_bulb_temperature (float):
            A dry bulb temperature in degrees Celsius.
        relative_humidity (float):
            A relative humidity in percent (0-100).
        evaporative_cooling_effectiveness (float):
            The evaporative cooling effectiveness. This should be a value between 0 (no effect)
            and 1 (saturated air).
        atmospheric_pressure (float, optional):
            A pressure in Pa. Default is pressure at sea level (101325 Pa).

    Returns:
        effective_dry_bulb_temperature, effective_relative_humidity (list[float]):
            A list of two values for the effective dry bulb temperature and relative humidity.
    """

    if atmospheric_pressure is None:
        atmospheric_pressure = 101325

    wet_bulb_temperature = wet_bulb_from_db_rh(
        dry_bulb_temperature, relative_humidity, atmospheric_pressure
    )

    new_dbt = dry_bulb_temperature - (
        (dry_bulb_temperature - wet_bulb_temperature)
        * evaporative_cooling_effectiveness
    )
    new_rh = (
        relative_humidity * (1 - evaporative_cooling_effectiveness)
    ) + evaporative_cooling_effectiveness * 100

    if new_rh > 100:
        new_rh = 100
        new_dbt = wet_bulb_temperature

    return [new_dbt, new_rh]


@bhom_analytics()
def evaporative_cooling_effect_collection(
    epw: EPW, evaporative_cooling_effectiveness: float = 0.3
) -> list[HourlyContinuousCollection]:
    """Calculate the effective DBT and RH considering effects of evaporative cooling.

    Args:
        epw (EPW): A ladybug EPW object.
        evaporative_cooling_effectiveness (float, optional):
            The proportion of difference between DBT and WBT by which to adjust DBT.
            Defaults to 0.3 which equates to 30% effective evaporative cooling, roughly that
            of Misting.

    Returns:
        list[HourlyContinuousCollection]:
            Adjusted dry-bulb temperature and relative humidity collections incorporating
            evaporative cooling effect.
    """

    if (evaporative_cooling_effectiveness > 1) or (
        evaporative_cooling_effectiveness < 0
    ):
        raise ValueError("evaporative_cooling_effectiveness must be between 0 and 1.")

    wbt = HourlyContinuousCollection.compute_function_aligned(
        wet_bulb_from_db_rh,
        [
            epw.dry_bulb_temperature,
            epw.relative_humidity,
            epw.atmospheric_station_pressure,
        ],
        WetBulbTemperature(),
        "C",
    )
    dbt = epw.dry_bulb_temperature.duplicate()
    dbt = dbt - ((dbt - wbt) * evaporative_cooling_effectiveness)
    dbt.header.metadata[
        "evaporative_cooling"
    ] = f"{evaporative_cooling_effectiveness:0.0%}"

    rh = epw.relative_humidity.duplicate()
    rh = (rh * (1 - evaporative_cooling_effectiveness)) + (
        evaporative_cooling_effectiveness * 100
    )
    rh.header.metadata[
        "evaporative_cooling"
    ] = f"{evaporative_cooling_effectiveness:0.0%}"

    return [dbt, rh]

@bhom_analytics()
def month_hour_binned_series(
    series: pd.Series,
    month_bins: tuple[tuple[int]] = None,
    hour_bins: tuple[tuple[int]] = None,
    month_labels: tuple[str] = None,
    hour_labels: tuple[str] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Bin a series by hour and month.

    Args:
        series (pd.Series):
            A series with a datetime index.
        hour_bins (tuple[tuple[int]], optional):
            A list of lists of hours to bin by. Defaults to None which bins into the default_time_analysis_periods().
        month_bins (tuple[tuple[int]], optional):
            A list of lists of months to bin by. Defaults to None which bins into default_month_analysis_periods.
        hour_labels (list[str], optional):
            A list of labels to use for the hour bins. Defaults to None which just lists the hours in each bin.
        month_labels (list[str], optional):
            A list of labels to use for the month bins. Defaults to None which just lists the months in each bin.
        agg (str, optional):
            The aggregation method to use. Can be either "min", "mean", "median", "max" or "sum". Defaults to "mean".

    Returns:
        time_binned_df (pd.DataFrame):
            A dataframe with the binned series.
    """

    # check that input dtype is pd.Series
    if not isinstance(series, pd.Series):
        raise ValueError("The series must be a pandas Series.")

    # check that the series is time indexed
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series must be a time series")

    # check that the series is not empty
    if series.empty:
        raise ValueError("The series cannot be empty")

    # check that the series contains 12-months worth of data at least
    if len(np.unique(series.index.month)) < 12:
        raise ValueError("The series must contain at least 12-months")

    # check that the series has at least 24-values per day
    if series.groupby(series.index.day_of_year).count().min() < 24:
        raise ValueError("The series must contain at least 24-values per day")

    # create generic month/hour sets for binning, from existing defaults if no input
    if month_bins is None:
        _months = np.arange(1, 13, 1)
        month_bins = []
        for ap in default_month_analysis_periods():
            if ap.st_month == ap.end_month:
                res = (ap.st_month,)
            else:
                length = ap.end_month - ap.st_month
                res = tuple(np.roll(_months, -ap.st_month + 1)[: length + 1])
            month_bins.append(res)
        month_bins = tuple(month_bins)
    if hour_bins is None:
        _hours = np.arange(0, 24)
        hour_bins = []
        for ap in default_hour_analysis_periods():
            if ap.st_hour == ap.end_hour:
                res = (ap.st_hour,)
            else:
                length = ap.end_hour - ap.st_hour
                res = tuple(np.roll(_hours, -ap.st_hour)[: length + 1])
            hour_bins.append(res)
        hour_bins = tuple(hour_bins)

    # check for contiguity of time periods
    flat_hours = [item for sublist in hour_bins for item in sublist]
    flat_months = [item for sublist in month_bins for item in sublist]
    if (max(flat_hours) != 23) or min(flat_hours) != 0:
        raise ValueError("hour_bins hours must be in the range 0-23")
    if (max(flat_months) != 12) or min(flat_months) != 1:
        raise ValueError("month_bins hours must be in the range 1-12")
    # check for duplicates
    if len(set(flat_hours)) != len(flat_hours):
        raise ValueError("hour_bins hours must not contain duplicates")
    if len(set(flat_months)) != len(flat_months):
        raise ValueError("month_bins hours must not contain duplicates")
    if (set(flat_hours) != set(list(range(24)))) or (len(set(flat_hours)) != 24):
        raise ValueError("Input hour_bins does not contain all hours of the day")
    if (set(flat_months) != set(list(range(1, 13, 1)))) or (
        len(set(flat_months)) != 12
    ):
        raise ValueError("Input month_bins does not contain all months of the year")

    # create index/column labels
    if month_labels:
        if len(month_labels) != len(month_bins):
            raise ValueError("month_labels must be the same length as month_bins")
        col_labels = month_labels
    else:
        col_labels = []
        for months in month_bins:
            if len(months) == 1:
                col_labels.append(calendar.month_abbr[months[0]])
            else:
                col_labels.append(
                    f"{calendar.month_abbr[months[0]]} to {calendar.month_abbr[months[-1]]}"
                )
    if hour_labels:
        if len(hour_labels) != len(hour_bins):
            raise ValueError("hour_labels must be the same length as hour_bins")
        row_labels = hour_labels
    else:
        row_labels = [f"{i[0]:02d}:00 ≤ t < {i[-1] + 1:02d}:00" for i in hour_bins]

    # create indexing bins
    values = []
    for months in month_bins:
        month_mask = series.index.month.isin(months)
        inner_values = []
        for hours in hour_bins:
            mask = series.index.hour.isin(hours) & month_mask
            aggregated = series.loc[mask].agg(agg)
            inner_values.append(aggregated)
        values.append(inner_values)
    df = pd.DataFrame(values, index=col_labels, columns=row_labels).T

    return df


def sunrise_sunset(location: Location) -> pd.DataFrame():
    """Calculate sunrise and sunset times for a given location and year. Includes
    civil, nautical and astronomical twilight.

    Args:
        location (Location): The location to calculate sunrise and sunset for.

    Returns:
        pd.DataFrame: A DataFrame with sunrise and sunset times for each day of the year.
    """

    idx = pd.date_range("2017-01-01", "2017-12-31", freq="D")
    sp = Sunpath.from_location(location)
    df = pd.DataFrame(
        [
            {
                **sp.calculate_sunrise_sunset_from_datetime(
                    lb_datetime_from_datetime(ix), depression=0.5334
                ),  # actual sunrise/set
                **{
                    f"civil {k}".replace("sunrise", "twilight start").replace(
                        "sunset", "twilight end"
                    ): v
                    for k, v in sp.calculate_sunrise_sunset_from_datetime(
                        lb_datetime_from_datetime(ix), depression=6
                    ).items()
                    if k != "noon"
                },  # civil twilight
                **{
                    f"nautical {k}".replace("sunrise", "twilight start").replace(
                        "sunset", "twilight end"
                    ): v
                    for k, v in sp.calculate_sunrise_sunset_from_datetime(
                        lb_datetime_from_datetime(ix), depression=12
                    ).items()
                    if k != "noon"
                },  # nautical twilight
                **{
                    f"astronomical {k}".replace("sunrise", "twilight start").replace(
                        "sunset", "twilight end"
                    ): v
                    for k, v in sp.calculate_sunrise_sunset_from_datetime(
                        lb_datetime_from_datetime(ix), depression=18
                    ).items()
                    if k != "noon"
                },  # astronomical twilight
            }
            for ix in idx
        ],
        index=idx,
    )
    return df[
        [
            "astronomical twilight start",
            "nautical twilight start",
            "civil twilight start",
            "sunrise",
            "noon",
            "sunset",
            "civil twilight end",
            "nautical twilight end",
            "astronomical twilight end",
        ]
    ]