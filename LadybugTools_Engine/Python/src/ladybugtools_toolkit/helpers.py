from __future__ import annotations

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
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.datatype.temperature import WetBulbTemperature
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection, Location
from ladybug.psychrometrics import wet_bulb_from_db_rh
from ladybug.skymodel import (
    calc_horizontal_infrared,
    calc_sky_temperature,
    estimate_illuminance_from_irradiance,
    get_extra_radiation,
    zhang_huang_solar,
    zhang_huang_solar_split,
)
from ladybug.sunpath import Sunpath
from matplotlib.ticker import FuncFormatter
from scipy.stats import weibull_min
from tqdm import tqdm


@FuncFormatter
def ZeroPadPercentFormatter(x: float) -> str:
    """A matplotlib formatter for percentages that pads with zeros.

    Args:
        x (float): The value to be formatted.

    Returns:
        str: The formatted string.
    """

    return f"{x:5.0%}"


def default_hour_analysis_periods() -> List[AnalysisPeriod]:
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


def default_month_analysis_periods() -> List[AnalysisPeriod]:
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


def default_combined_analysis_periods() -> List[AnalysisPeriod]:
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


def default_analysis_periods() -> List[AnalysisPeriod]:
    """A set of generic Analysis Period objects, spanning all predefined combinations of time of ady and month of year."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        aps = [
            AnalysisPeriod(),
        ]
        aps.extend(default_month_analysis_periods())
        aps.extend(default_hour_analysis_periods())
        aps.extend(default_combined_analysis_periods())

    return aps


def chunks(lst: List[Any], chunksize: int):
    """Partition an iterable into lists of length "chunksize".

    Args:
        lst (List[Any]): The list to be partitioned.
        chunksize (int): The size of each partition.

    Yields:
        List[Any]: A list of length "chunksize".
    """
    for i in range(0, len(lst), chunksize):
        yield lst[i : i + chunksize]


def scrape_weather(
    station: str,
    start_date: str = "1970-01-01",
    end_date: str = None,
    interpolate: bool = False,
    resample: bool = False,
) -> pd.DataFrame:
    """Scrape historic data from global airport weather stations using their ICAO codes
        (https://en.wikipedia.org/wiki/List_of_airports_by_IATA_and_ICAO_code)

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
    uri = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={station}&year1={start_date.year}&month1={start_date.month}&day1={start_date.day}&year2={end_date.year}&month2={end_date.month}&day2={end_date.day}&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=null&trace=null&direct=no&data=tmpc&data=dwpc&data=relh&data=drct&data=sknt&data=alti&data=p01m&data=vsby&data=skyc1&data=skyc2&data=skyc3"
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
        df = df.resample("30T").mean()

    return df


def rolling_window(array: List[Any], window: int):
    """Throwaway function here to roll a window along a list.

    Args:
        array (List[Any]):
            A 1D list of some kind.
        window (int):
            The size of the window to apply to the list.

    Example:
        For an input list like [0, 1, 2, 3, 4, 5, 6, 7, 8],
        returns [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]

    Returns:
        List[List[Any]]:
            The resulting, "windowed" list.
    """

    if window > len(array):
        raise ValueError("Array length must be larger than window size.")

    a: np.ndarray = np.array(array)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)  # pylint: disable=unsubscriptable-object
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def proximity_decay(
    value: float,
    distance_to_value: float,
    max_distance: float,
    decay_method: str = "linear",
) -> float:
    """Calculate the "decayed" value based on proximity (up to a maximum distance).

    Args:
        value (float):
            The value to be distributed.
        distance_to_value (float):
            A distance at which to return the magnitude.
        max_distance (float):
            The maximum distance to which magnitude is to be distributed. Beyond this, the input
            value is 0.
        decay_method (str, optional):
            A type of distribution (the shape of the distribution profile). Defaults to "linear".

    Returns:
        float:
            The value at the given distance.
    """

    distance_to_value = np.interp(distance_to_value, [0, max_distance], [0, 1])

    if decay_method == "linear":
        return (1 - distance_to_value) * value
    if decay_method == "parabolic":
        return (-(distance_to_value**2) + 1) * value
    if decay_method == "sigmoid":
        return (1 - (0.5 * (np.sin(distance_to_value * np.pi - np.pi / 2) + 1))) * value

    raise ValueError(f"Unknown curve type: {decay_method}")


def timedelta_tostring(time_delta: timedelta) -> str:
    """timedelta objects don't have a nice string representation, so this function converts them.

    Args:
        time_delta (datetime.timedelta):
            The timedelta object to convert.
    Returns:
        str:
            A string representation of the timedelta object.
    """
    s = time_delta.seconds
    hours, remainder = divmod(s, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}"


def decay_rate_smoother(
    series: pd.Series,
    difference_threshold: float = -10,
    transition_window: int = 4,
    ewm_span: float = 1.25,
) -> pd.Series:
    """Helper function that adds a decay rate to a time-series for values dropping significantly
        below the previous values.

    Args:
        series (pd.Series):
            The series to modify
        difference_threshold (float, optional):
            The difference between current/previous values which class as a "transition".
            Defaults to -10.
        transition_window (int, optional):
            The number of values after the "transition" within which an exponentially weighted mean
             should be applied. Defaults to 4.
        ewm_span (float, optional):
            The rate of decay. Defaults to 1.25.

    Returns:
        pd.Series:
            A modified series
    """

    # Find periods of major transition (where values vary significantly)
    transition_index = series.diff() < difference_threshold

    # Get boolean index for all periods within window from the transition indices
    ewm_mask = []
    n = 0
    for i in transition_index:
        if i:
            n = 0
        if n < transition_window:
            ewm_mask.append(True)
        else:
            ewm_mask.append(False)
        n += 1

    # Run an EWM to get the smoothed values following changes to values
    ewm_smoothed: pd.Series = series.ewm(span=ewm_span).mean()

    # Choose from ewm or original values based on ewm mask
    new_series = ewm_smoothed.where(ewm_mask, series)

    return new_series


def cardinality(direction_angle: float, directions: int = 16):
    """Returns the cardinal orientation of a given angle, where that angle is related to north at
        0 degrees.
    Args:
        direction_angle (float):
            The angle to north in degrees (+Ve is interpreted as clockwise from north at 0.0
            degrees).
        directions (int):
            The number of cardinal directions into which angles shall be binned (This value should
            be one of 4, 8, 16 or 32, and is centred about "north").
    Returns:
        int:
            The cardinal direction the angle represents.
    """

    if direction_angle > 360 or direction_angle < 0:
        raise ValueError(
            "The angle entered is beyond the normally expected range for an orientation in degrees."
        )

    cardinal_directions = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        32: [
            "N",
            "NbE",
            "NNE",
            "NEbN",
            "NE",
            "NEbE",
            "ENE",
            "EbN",
            "E",
            "EbS",
            "ESE",
            "SEbE",
            "SE",
            "SEbS",
            "SSE",
            "SbE",
            "S",
            "SbW",
            "SSW",
            "SWbS",
            "SW",
            "SWbW",
            "WSW",
            "WbS",
            "W",
            "WbN",
            "WNW",
            "NWbW",
            "NW",
            "NWbN",
            "NNW",
            "NbW",
        ],
    }

    if directions not in cardinal_directions:
        raise ValueError(
            f'The input "directions" must be one of {list(cardinal_directions.keys())}.'
        )

    val = int((direction_angle / (360 / directions)) + 0.5)

    arr = cardinal_directions[directions]

    return arr[(val % directions)]


def angle_from_cardinal(cardinal_direction: str) -> float:
    """
    For a given cardinal direction, return the corresponding angle in degrees.

    Args:
        cardinal_direction (str):
            The cardinal direction.
    Returns:
        float:
            The angle associated with the cardinal direction.
    """
    cardinal_directions = [
        "N",
        "NbE",
        "NNE",
        "NEbN",
        "NE",
        "NEbE",
        "ENE",
        "EbN",
        "E",
        "EbS",
        "ESE",
        "SEbE",
        "SE",
        "SEbS",
        "SSE",
        "SbE",
        "S",
        "SbW",
        "SSW",
        "SWbS",
        "SW",
        "SWbW",
        "WSW",
        "WbS",
        "W",
        "WbN",
        "WNW",
        "NWbW",
        "NW",
        "NWbN",
        "NNW",
        "NbW",
    ]
    if cardinal_direction not in cardinal_directions:
        raise ValueError(f"{cardinal_direction} is not a known cardinal_direction.")
    angles = np.arange(0, 360, 11.25)

    lookup = dict(zip(cardinal_directions, angles))

    return lookup[cardinal_direction]


def angle_from_north(vector: List[float]) -> float:
    """For an X, Y vector, determine the clockwise angle to north at [0, 1].

    Args:
        vector (List[float]):
            A vector of length 2.

    Returns:
        float:
            The angle between vector and north in degrees clockwise from [0, 1].
    """
    north = [0, 1]
    angle1 = np.arctan2(*north[::-1])
    angle2 = np.arctan2(*vector[::-1])
    return np.rad2deg((angle1 - angle2) % (2 * np.pi))


def angle_to_vector(_angle_from_north: float) -> List[float]:
    """Return the X, Y vector from of an angle from north at 0-degrees."""
    if (_angle_from_north > 360) or (_angle_from_north < 0):
        raise ValueError(
            "angle_from_north must be between 0 and 360 degrees (inclusive)."
        )

    _angle_from_north = np.radians(_angle_from_north)

    return np.sin(_angle_from_north), np.cos(_angle_from_north)


def sanitise_string(string: str) -> str:
    """Sanitise a string so that only path-safe characters remain."""
    keep_characters = r"[^.A-Za-z0-9_-]"
    return re.sub(keep_characters, "_", string).replace("__", "_").rstrip()


def stringify_df_header(columns: List[Any]) -> List[str]:
    """Convert a list of objects into their string representations. This method is mostly used for making DataFrames parqeut serialisable.

    Args:
        columns (List[Any]): The list of objects to be converted.

    Returns:
        List[str]: The list of strings.
    """

    return [str(i) for i in columns]


def unstringify_df_header(columns: List[str]) -> List[Any]:
    """Convert a list of strings into a set of objects capable of being used as DataFrame column headers.

    Args:
        columns (List[str]): The list of strings to be converted.

    Returns:
        List[Any]: The list of objects.
    """

    evaled = []
    for i in columns:
        try:
            evaled.append(eval(i))  # pylint: disable=eval-used
        except NameError:
            evaled.append(i)

    if all("(" in i for i in str(evaled)):
        return pd.MultiIndex.from_tuples(evaled)
    return evaled


def write_parquet(data: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write a parquet file, and convert columns into strings.

    Args:
        data (pd.DataFrame): The data to be written.
        path (Union[str, Path]): The path to the parquet file.

    Returns:
        Path: The path to the parquet file.
    """
    path = Path(path)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(i) for i in data.columns]

    print(f"Writing {path.as_posix()}")

    data.to_parquet(path)

    return path


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a parquet file, and reinterpret columns as their automatically assigned datatype.

    Args:
        path (Union[str, Path]): The path to the parquet file.

    Returns:
        pd.DataFrame: A pandas DataFrame.
    """

    path = Path(path)

    print(f"Reading {path.as_posix()}")

    df = pd.read_parquet(path)
    try:
        df.columns = pd.MultiIndex.from_tuples([eval(i) for i in df.columns])
    except:
        print("yo")

    return df


def store_dataset(
    dataframe: pd.DataFrame, target_path: Path, downcast: bool = True
) -> Path:
    """Serialise a pandas DataFrame as a parquet file, including pre-processing to ensure it is serialisable, and optional downcasting to samller dtypes.

    Args:
        dataframe (pd.DataFrame):
            The dataframe to be serialised
        target_path (Path):
            The target file for storage.
        downcast (bool, optional):
            Optinal downcasting to reduce dataframe dtype complexity. Defaults to True.

    Returns:
        Path:
            The path to the stored dataset.
    """

    target_path = Path(target_path)

    if "parquet" not in target_path.suffix:
        raise ValueError('This method only currently works for "*.parquet" files.')

    if downcast:
        # downcast dataframe type to save on storage
        for col, _type in dataframe.dtypes.items():
            if _type == "float64":
                dataframe[col] = dataframe[col].astype(np.float32)
            if _type == "int64":
                dataframe[col] = dataframe[col].astype(np.int16)

    # prepare dataframe for storage as parquet
    dataframe.columns = stringify_df_header(dataframe.columns)

    # store dataframe
    dataframe.to_parquet(target_path, compression="snappy")

    return target_path


def load_dataset(target_path: Path, upcast: bool = True) -> pd.DataFrame:
    """Read a stored dataset, including upcasting to float64.

    Args:
        target_path (Path):
            The dataset path to be loaded.
        upcast (bool, optional):
            Optional upcasting of data to enable downstream calculations without issues!

    Returns:
        pd.DataFrame:
            The loaded dataset as a dataframe.
    """

    df = pd.read_parquet(target_path)

    if upcast:
        for col, _type in dict(df.dtypes).items():
            if _type == "float32":
                df[col].astype(np.float64)
            elif _type in ["int32", "int16", "int8"]:
                df[col].astype(np.int64)

    df.columns = unstringify_df_header(df.columns)

    return df


class OpenMeteoVariable(Enum):
    """An enumeration of variables available from OpenMeteo, and their metadata for handling returned values."""

    TEMPERATURE_2M = auto()
    DEWPOINT_2M = auto()
    RELATIVEHUMIDITY_2M = auto()
    SURFACE_PRESSURE = auto()
    SHORTWAVE_RADIATION = auto()
    DIRECT_RADIATION = auto()
    DIFFUSE_RADIATION = auto()
    WINDDIRECTION_10M = auto()
    WINDSPEED_10M = auto()
    CLOUDCOVER = auto()
    PRECIPITATION = auto()
    RAIN = auto()
    SNOWFALL = auto()
    CLOUDCOVER_LOW = auto()
    CLOUDCOVER_MID = auto()
    CLOUDCOVER_HIGH = auto()
    DIRECT_NORMAL_IRRADIANCE = auto()
    WINDSPEED_100M = auto()
    WINDDIRECTION_100M = auto()
    WINDGUSTS_10M = auto()
    ET0_FAO_EVAPOTRANSPIRATION = auto()
    VAPOR_PRESSURE_DEFICIT = auto()
    SOIL_TEMPERATURE_0_TO_7CM = auto()
    SOIL_TEMPERATURE_7_TO_28CM = auto()
    SOIL_TEMPERATURE_28_TO_100CM = auto()
    SOIL_TEMPERATURE_100_TO_255CM = auto()
    SOIL_MOISTURE_0_TO_7CM = auto()
    SOIL_MOISTURE_7_TO_28CM = auto()
    SOIL_MOISTURE_28_TO_100CM = auto()
    SOIL_MOISTURE_100_TO_255CM = auto()

    @staticmethod
    def __properties__() -> Dict[str, Dict[str, Union[str, float]]]:
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
            OpenMeteoVariable.DIRECT_RADIATION.value: {
                "openmeteo_name": "direct_radiation",
                "openmeteo_unit": "W/m²",
                "target_name": "Direct Horizontal Radiation",
                "target_unit": "Wh/m2",
                "target_multiplier": 1,
            },
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
            OpenMeteoVariable.SNOWFALL.value: {
                "openmeteo_name": "snowfall",
                "openmeteo_unit": "cm",
                "target_name": "Snow depth",
                "target_unit": "cm",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.CLOUDCOVER_LOW.value: {
                "openmeteo_name": "cloudcover_low",
                "openmeteo_unit": "%",
                "target_name": "Cloud Cover @<2km",
                "target_unit": "tenths",
                "target_multiplier": 0.1,
            },
            OpenMeteoVariable.CLOUDCOVER_MID.value: {
                "openmeteo_name": "cloudcover_mid",
                "openmeteo_unit": "%",
                "target_name": "Cloud Cover @2-6km",
                "target_unit": "tenths",
                "target_multiplier": 0.1,
            },
            OpenMeteoVariable.CLOUDCOVER_HIGH.value: {
                "openmeteo_name": "cloudcover_high",
                "openmeteo_unit": "%",
                "target_name": "Cloud Cover @>6km",
                "target_unit": "tenths",
                "target_multiplier": 0.1,
            },
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
            OpenMeteoVariable.WINDGUSTS_10M.value: {
                "openmeteo_name": "windgusts_10m",
                "openmeteo_unit": "km/h",
                "target_name": "Wind Gusts @10m",
                "target_unit": "m/s",
                "target_multiplier": 1 / 3.6,
            },
            OpenMeteoVariable.ET0_FAO_EVAPOTRANSPIRATION.value: {
                "openmeteo_name": "et0_fao_evapotranspiration",
                "openmeteo_unit": "mm",
                "target_name": "Evapotranspiration",
                "target_unit": "mm/inch",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.VAPOR_PRESSURE_DEFICIT.value: {
                "openmeteo_name": "vapor_pressure_deficit",
                "openmeteo_unit": "kPa",
                "target_name": "Vapor Pressure Deficit",
                "target_unit": "Pa",
                "target_multiplier": 0.001,
            },
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
            OpenMeteoVariable.SOIL_MOISTURE_0_TO_7CM.value: {
                "openmeteo_name": "soil_moisture_0_to_7cm",
                "openmeteo_unit": "m³/m³",
                "target_name": "Soil Moisture @0-7cm",
                "target_unit": "fraction",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SOIL_MOISTURE_7_TO_28CM.value: {
                "openmeteo_name": "soil_moisture_7_to_28cm",
                "openmeteo_unit": "m³/m³",
                "target_name": "Soil Moisture @7-28cm",
                "target_unit": "fraction",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SOIL_MOISTURE_28_TO_100CM.value: {
                "openmeteo_name": "soil_moisture_28_to_100cm",
                "openmeteo_unit": "m³/m³",
                "target_name": "Soil Moisture @28-100cm",
                "target_unit": "fraction",
                "target_multiplier": 1,
            },
            OpenMeteoVariable.SOIL_MOISTURE_100_TO_255CM.value: {
                "openmeteo_name": "soil_moisture_100_to_255cm",
                "openmeteo_unit": "m³/m³",
                "target_name": "Soil Moisture @100-255cm",
                "target_unit": "fraction",
                "target_multiplier": 1,
            },
        }

    @classmethod
    def from_string(cls, name: str) -> OpenMeteoVariable:
        """."""
        d = {
            "temperature_2m": cls.TEMPERATURE_2M,
            "dewpoint_2m": cls.DEWPOINT_2M,
            "relativehumidity_2m": cls.RELATIVEHUMIDITY_2M,
            "surface_pressure": cls.SURFACE_PRESSURE,
            "shortwave_radiation": cls.SHORTWAVE_RADIATION,
            "direct_radiation": cls.DIRECT_RADIATION,
            "diffuse_radiation": cls.DIFFUSE_RADIATION,
            "winddirection_10m": cls.WINDDIRECTION_10M,
            "windspeed_10m": cls.WINDSPEED_10M,
            "cloudcover": cls.CLOUDCOVER,
            "precipitation": cls.PRECIPITATION,
            "rain": cls.RAIN,
            "snowfall": cls.SNOWFALL,
            "cloudcover_low": cls.CLOUDCOVER_LOW,
            "cloudcover_mid": cls.CLOUDCOVER_MID,
            "cloudcover_high": cls.CLOUDCOVER_HIGH,
            "direct_normal_irradiance": cls.DIRECT_NORMAL_IRRADIANCE,
            "windspeed_100m": cls.WINDSPEED_100M,
            "winddirection_100m": cls.WINDDIRECTION_100M,
            "windgusts_10m": cls.WINDGUSTS_10M,
            "et0_fao_evapotranspiration": cls.ET0_FAO_EVAPOTRANSPIRATION,
            "vapor_pressure_deficit": cls.VAPOR_PRESSURE_DEFICIT,
            "soil_temperature_0_to_7cm": cls.SOIL_TEMPERATURE_0_TO_7CM,
            "soil_temperature_7_to_28cm": cls.SOIL_TEMPERATURE_7_TO_28CM,
            "soil_temperature_28_to_100cm": cls.SOIL_TEMPERATURE_28_TO_100CM,
            "soil_temperature_100_to_255cm": cls.SOIL_TEMPERATURE_100_TO_255CM,
            "soil_moisture_0_to_7cm": cls.SOIL_MOISTURE_0_TO_7CM,
            "soil_moisture_7_to_28cm": cls.SOIL_MOISTURE_7_TO_28CM,
            "soil_moisture_28_to_100cm": cls.SOIL_MOISTURE_28_TO_100CM,
            "soil_moisture_100_to_255cm": cls.SOIL_MOISTURE_100_TO_255CM,
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


def scrape_openmeteo(
    latitude: float,
    longitude: float,
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    variables: Tuple[OpenMeteoVariable] = None,
    convert_units: bool = False,
) -> pd.DataFrame:
    """Obtain historic hourly data from Open-Meteo.
    https://open-meteo.com/en/docs/historical-weather-api

    Args:
        latitude (float):
            The latitude of the target site, in degrees.
        longitude (float):
            The longitude of the target site, in degrees.
        start_date (Union[datetime, str]):
            The start-date from which records will be obtained.
        end_date (Union[datetime, str]):
            The end-date beyond which records will be ignored.
        variables (Tuple[OpenMeteoVariable]):
            A list of variables to query. If None, then all variables will be queried.
        convert_units (bool, optional):
            Convert units output into more common units, and rename headers accordingly.

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
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date > end_date:
        raise ValueError("The start_date must be before the end_date.")

    if variables is None:
        variables = tuple(OpenMeteoVariable)
    else:
        if not all(isinstance(val, OpenMeteoVariable) for val in variables):
            raise ValueError(
                "All values in the variables tuple must be of type OpenMeteoVariable."
            )

    # construct query string
    var_strings = ",".join([i.openmeteo_name for i in variables])
    query_string = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={start_date:%Y-%m-%d}&end_date={end_date:%Y-%m-%d}&hourly={var_strings}"

    # request data
    with urllib.request.urlopen(query_string) as url:
        data = json.load(url)

    # convert resultant data to dataframe
    headers = [f"{k} ({v})" for (k, v) in data["hourly_units"].items()]
    values = [v for _, v in data["hourly"].items()]
    df = pd.DataFrame(np.array(values).T, columns=headers)
    df = df.set_index(df.columns[0])
    df.index.name = None
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    if not convert_units:
        return df

    new_df = []
    for i in OpenMeteoVariable:
        try:
            new_df.append(
                i.convert(df[i.openmeteo_table_name]).rename(i.target_table_name)
            )
        except KeyError:
            pass
    return pd.concat(new_df, axis=1)


def get_soil_temperatures(
    latitude: float,
    longitude: float,
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    include_dbt: bool = False,
) -> pd.DataFrame:
    """Query Open-Meteo for soil temperature data.

    Args:
        latitude (float): The latitude of the target site, in degrees.
        longitude (float): The longitude of the target site, in degrees.
        start_date (Union[datetime, str]): The start-date from which records will be obtained.
        end_date (Union[datetime, str]): The end-date beyond which records will be ignored.
        include_dbt (bool, optional): Include dry bulb temperature in the output. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing scraped data.
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


def weibull_directional(
    binned_data: Dict[Tuple[float, float], List[float]]
) -> pd.DataFrame:
    """Calculate the weibull coefficients for a given set of binned data in the form {(low, high): [speeds], (low, high): [speeds]}, binned by the number of directions specified.
    Args:
        binned_data (Dict[Tuple[float, float], List[float]]):
            A dictionary of binned wind speed data.
    Returns:
        pd.DataFrame:
            A DataFrame with (direction_bin_low, direction_bin_high) as index, and weibull coefficients as columns.
    """
    d = {}
    for (low, high), speeds in tqdm(
        binned_data.items(), desc="Calculating Weibull shape parameters"
    ):
        d[(low, high)] = weibull_pdf(speeds)

    return pd.DataFrame.from_dict(d, orient="index", columns=["k", "loc", "c"])


def weibull_pdf(wind_speeds: List[float]) -> Tuple[float]:
    """Estimate the two-parameter Weibull parameters for a set of wind speeds.

    Args:
        wind_speeds (List[float]):
            A list of wind speeds.

    Returns:
        k (float):
            Shape parameter.
        loc (float):
            Location parameter.
        c (float):
            Scale parameter.
    """

    ws = np.array(wind_speeds)
    ws = ws[ws != 0]
    ws = ws[~np.isnan(ws)]
    if ws.min() < 0:
        raise ValueError("Wind speeds must be positive.")
    try:
        return weibull_min.fit(ws)
    except ValueError as exc:
        warnings.warn(f"Not enough data to calculate Weibull parameters.\n{exc}")
        return (np.nan, np.nan, np.nan)  # type: ignore


def circular_weighted_mean(angles: List[float], weights: List[float] = None):
    """Get the average angle from a set of weighted angles.

    Args:
        angles (List[float]):
            A collection of equally weighted wind directions, in degrees from North (0).
        weights (List[float]):
            A collection of weights, which must sum to 1. Defaults to None which will equally weight all angles.

    Returns:
        float:
            An average wind direction.
    """
    angles = np.array(angles)
    if weights is None:
        weights = np.ones_like(angles) / len(angles)
    weights = np.array(weights)

    if angles.shape != weights.shape:
        raise ValueError("weights must be the same size as angles.")

    if np.any(angles < 0) or np.any(angles > 360):
        raise ValueError("Input angles exist outside of expected range (0-360).")

    weights = np.array(weights)
    weights = weights / sum(weights)

    x = y = 0.0
    for angle, weight in zip(angles, weights):
        x += np.cos(np.radians(angle)) * weight
        y += np.sin(np.radians(angle)) * weight

    mean = np.degrees(np.arctan2(y, x))

    if mean < 0:
        mean = 360 + mean

    return mean


def wind_direction_average(angles: List[float]) -> float:
    """Get the average wind direction from a set of wind directions.

    Args:
        angles (List[float]):
            A collection of equally weighted wind directions, in degrees from North (0).

    Returns:
        float:
            An average angle.
    """

    angles = np.array(angles)  # type: ignore

    if np.any(angles < 0) or np.any(angles > 360):  # type: ignore
        raise ValueError("Input angles exist outside of expected range (0-360).")

    if len(angles) == 0:
        return np.NaN

    angles = np.radians(angles)  # type: ignore

    average_angle = np.round(
        np.arctan2(
            (1 / len(angles) * np.sin(angles)).sum(),
            (1 / len(angles) * np.cos(angles)).sum(),
        ),
        2,
    )

    if average_angle < 0:
        average_angle = (np.pi * 2) - -average_angle

    return np.degrees(average_angle)


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

    Keyword Args:
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
    """Estimate the dry-bulb temperature at a given height from a referenced
        dry-bulb temperature at another height.

    Args:
        reference_value (float):
            The temperature to translate.
        reference_height (float):
            The height of the reference temperature.
        target_height (float):
            The height to translate the reference temperature towards.

    Keyword Args:
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

    if (target_height > 8000) or (reference_height > 8000):
        warnings.warn(
            "The heights input into this calculation exist partially above the egde of the troposphere. This method is only valid below 8000m."
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
    """Calculate the radiation at a given height, given a reference radiation and height.

    References:
        Armel Oumbe, Lucien Wald. A parameterisation of vertical profile of solar irradiance for correcting solar fluxes for changes in terrain elevation. Earth Observation and Water Cycle Science Conference, Nov 2009, Frascati, Italy. pp.S05.

    Args:
        reference_value (float):
            The radiation at the reference height.
        target_height (float):
            The height at which the radiation is required, in m.
        reference_height (float, optional):
            The height at which the reference radiation was measured.

    Keyword Arguments:
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


def validate_timeseries(
    obj: Any,
    is_annual: bool = False,
    is_hourly: bool = False,
    is_contiguous: bool = False,
) -> None:
    """Check if the input object is a pandas Series, and has a datetime index.

    Args:
        obj (Any):
            The object to check.
        is_annual (bool, optional):
            If True, check that the series is annual. Defaults to False.
        is_hourly (bool, optional):
            If True, check that the series is hourly. Defaults to False.
        is_contiguous (bool, optional):
            If True, check that the series is contiguous. Defaults to False.

    Raises:
        TypeError: If the object is not a pandas Series.
        TypeError: If the series does not have a datetime index.
        ValueError: If the series is not annual.
        ValueError: If the series is not hourly.
        ValueError: If the series is not contiguous.
    """
    if not isinstance(obj, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("series must have a datetime index")
    if is_annual:
        if (obj.index.day_of_year.nunique() != 365) or (
            obj.index.day_of_year.nunique() != 366
        ):
            raise ValueError("series is not annual")
    if is_hourly:
        if obj.index.hour.nunique() != 24:
            raise ValueError("series is not hourly")
    if is_contiguous:
        if not obj.index.is_monotonic_increasing:
            raise ValueError("series is not contiguous")


def evaporative_cooling_effect(
    dry_bulb_temperature: float,
    relative_humidity: float,
    evaporative_cooling_effectiveness: float,
    atmospheric_pressure: float = None,
) -> List[float]:
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
        effective_dry_bulb_temperature, effective_relative_humidity (List[float]):
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


def evaporative_cooling_effect_collection(
    epw: EPW, evaporative_cooling_effectiveness: float = 0.3
) -> List[HourlyContinuousCollection]:
    """Calculate the effective DBT and RH considering effects of evaporative cooling.

    Args:
        epw (EPW): A ladybug EPW object.
        evaporative_cooling_effectiveness (float, optional):
            The proportion of difference between DBT and WBT by which to adjust DBT.
            Defaults to 0.3 which equates to 30% effective evaporative cooling, roughly that
            of Misting.

    Returns:
        List[HourlyContinuousCollection]:
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


def remove_leap_days(
    pd_object: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    """A removal of all timesteps within a time-indexed pandas
    object where the day is the 29th of February."""

    if not isinstance(pd_object.index, pd.DatetimeIndex):
        raise ValueError("The object provided should be datetime-indexed.")

    mask = (pd_object.index.month == 2) & (pd_object.index.day == 29)

    return pd_object[~mask]


def month_hour_binned_series(
    series: pd.Series,
    month_bins: Tuple[Tuple[int]] = None,
    hour_bins: Tuple[Tuple[int]] = None,
    month_labels: Tuple[str] = None,
    hour_labels: Tuple[str] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Bin a series by hour and month.

    Args:
        series (pd.Series):
            A series with a datetime index.
        hour_bins (Tuple[Tuple[int]], optional):
            A list of lists of hours to bin by. Defaults to None which bins into the default_time_analysis_periods().
        month_bins (Tuple[Tuple[int]], optional):
            A list of lists of months to bin by. Defaults to None which bins into default_month_analysis_periods.
        hour_labels (List[str], optional):
            A list of labels to use for the hour bins. Defaults to None which just lists the hours in each bin.
        month_labels (List[str], optional):
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
    # cehck for duplicates
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

    idx = pd.date_range(f"2017-01-01", f"2017-12-31", freq="D")
    sp = Sunpath.from_location(location)
    df = pd.DataFrame(
        [
            {
                **sp.calculate_sunrise_sunset_from_datetime(
                    ix, depression=0.5334
                ),  # actual sunrise/set
                **{
                    f"civil {k}".replace("sunrise", "twilight start").replace(
                        "sunset", "twilight end"
                    ): v
                    for k, v in sp.calculate_sunrise_sunset_from_datetime(
                        ix, depression=6
                    ).items()
                    if k != "noon"
                },  # civil twilight
                **{
                    f"nautical {k}".replace("sunrise", "twilight start").replace(
                        "sunset", "twilight end"
                    ): v
                    for k, v in sp.calculate_sunrise_sunset_from_datetime(
                        ix, depression=12
                    ).items()
                    if k != "noon"
                },  # nautical twilight
                **{
                    f"astronomical {k}".replace("sunrise", "twilight start").replace(
                        "sunset", "twilight end"
                    ): v
                    for k, v in sp.calculate_sunrise_sunset_from_datetime(
                        ix, depression=18
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


def safe_filename(filename: str) -> str:
    """Remove all non-alphanumeric characters from a filename."""
    return "".join(
        [c for c in filename if c.isalpha() or c.isdigit() or c == " "]
    ).strip()
