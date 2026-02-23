"""
Methods for converting objects into Ladybug objects.
"""

import json
from datetime import datetime
from functools import singledispatch
from typing import Any, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.color import Color
from ladybug.datacollection import (
    DailyCollection,
    HourlyContinuousCollection,
    HourlyDiscontinuousCollection,
    MonthlyCollection,
    MonthlyPerHourCollection,
)
from ladybug.datacollectionimmutable import (
    DailyCollectionImmutable,
    HourlyContinuousCollectionImmutable,
    HourlyDiscontinuousCollectionImmutable,
    MonthlyCollectionImmutable,
    MonthlyPerHourCollectionImmutable,
)
from ladybug.datatype import TYPESDICT
from ladybug.datatype.generic import GenericType
from ladybug.dt import DateTime
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.location import Location
from matplotlib.colors import Colormap
from pvlib.location import Location as pvlib_location
from pvlib.location import lookup_altitude

from ..ladybug_extension.location import get_tzinfo
from ..ladybug_extension._mapping import _TIMESTEP_FREQUENCY_MAP
from ..bhom.logger import CONSOLE_LOGGER
from .to_colour import to_colour

@singledispatch
def to_ladybug(obj: Any) -> Any:
    """Convert a Pandas object to a Ladybug object."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to a ladybug object.")


@to_ladybug.register(Colormap)
def _(obj: Colormap) -> tuple[Color]:
    """Convert a matplotlib colormap into a ladybug Colorset."""
    N = 11  # number of colors to sample
    return tuple([to_colour(obj(i), fmt="ladybug") for i in np.linspace(0, 1, N)])


@to_ladybug.register(pd.Timestamp)
def _(obj: pd.Timestamp) -> DateTime:
    """Convert a pandas Timestamp to a Ladybug DateTime."""
    return DateTime(
        month=obj.month,
        day=obj.day,
        hour=obj.hour,
        minute=obj.minute,
        leap_year=obj.is_leap_year,
    )


@to_ladybug.register(pd.DatetimeIndex)
def _(
    obj: pd.DatetimeIndex,
) -> AnalysisPeriod:
    """Convert a pandas DatetimeIndex into a Ladybug AnalysisPeriod."""
    # ensure datetimes are sorted
    if any(np.diff(obj) < pd.Timedelta(0)):
        raise ValueError("datetimes must be in order, they are not currently.")

    # ensure at most, 2 years in the analysis period
    if len(obj.year.unique()) > 2:
        raise ValueError("datetimes span more than 2 years")

    # find the earliest datetime and get the st_month and st_day and st_hour
    st_month = obj[0].month
    st_day = obj[0].day
    st_hour = obj[0].hour

    # find the latest datetime and get the end_month and end_day and end_hour
    end_month = obj[-1].month
    end_day = obj[-1].day
    end_hour = obj[-1].hour

    # check for leap year
    leap = obj[0].is_leap_year or obj[-1].is_leap_year

    # find the frequency of the datetimes, by sampling
    frequencies = []
    for sme in np.lib.stride_tricks.sliding_window_view(obj, 3):
        freq = pd.infer_freq(sme)
        if freq is None:
            continue
        frequencies.append(freq)
    if len(frequencies) == 0:
        raise ValueError("Could not determine the frequency of the datetimes.")
    frequencies = set(frequencies)
    if len(frequencies) > 1:
        CONSOLE_LOGGER.warning(
            "datetimes have multiple frequencies, using the minimum frequency."
        )

    try:
        timestep = {v: k for k, v in _TIMESTEP_FREQUENCY_MAP.items()}[min(frequencies)]
    except KeyError:
        raise ValueError(
            "Could not determine the frequency of the datetimes. "
            f"Please ensure that the datetimes are evenly spaced and one of these frequencies: {list(_TIMESTEP_FREQUENCY_MAP.values())}."
        )

    d = {
        "st_month": st_month,
        "st_day": st_day,
        "st_hour": st_hour,
        "end_month": end_month,
        "end_day": end_day,
        "end_hour": end_hour,
        "timestep": timestep,
        "is_leap_year": leap,
    }

    return AnalysisPeriod.from_dict(d)


@to_ladybug.register(tuple)
def _(obj: tuple[str, str, str]) -> Header:
    """Convert a tuple of strings to a ladybug Header object.

    Note:
    - This method assigns a default analysis period. For use in other workflows, override that default!

    """
    if len(obj) != 3:
        raise ValueError("Header tuples must have exactly 3 elements.")

    # ensure all elements are strings
    if not all(isinstance(i, str) for i in obj):
        raise TypeError("All elements of the header tuple must be strings.")

    # create a new header
    try:
        # normal header as CSV string
        header = Header.from_csv_strings(
            csv_strings=obj, analysis_period=AnalysisPeriod()
        )
    except AssertionError:
        if " | " in obj[0]:
            # likely a Generic dtype, so something special needs to be done instead
            (
                _name,
                _unit,
                _min,
                _max,
                _abbreviation,
                _unit_descr,
                _point_in_time,
                _cumulative,
            ) = obj[0].split(" | ")
            data_type = GenericType(
                name=_name,
                unit=_unit,
                min=float(_min),
                max=float(_max),
                abbreviation=_abbreviation,
                unit_descr=None if _unit_descr == "None" else _unit_descr,
                point_in_time=_point_in_time == "True",
                cumulative=_cumulative == "True",
            )
            all_props = [p for prop in obj[2:] for p in prop.split(" | ")]
            metadata = {}
            for p in all_props:
                p_split = p.split(": ")
                metadata[p_split[0]] = p_split[1]
            header = Header(
                data_type=data_type,
                unit=obj[1],
                analysis_period=AnalysisPeriod(),
                metadata=metadata,
            )
        else:
            raise ValueError(
                "The tuple provided contains an unserializable set of information"
            )
    except IndexError:
        CONSOLE_LOGGER.warning(
            "No metadata found in header tuple, creating header without metadata."
        )
        data_type = TYPESDICT[obj[0]]()
        unit = obj[1]
        metadata = {}
        header = Header(
            data_type=data_type,
            unit=unit,
            analysis_period=AnalysisPeriod(),
            metadata=metadata,
        )

    # attempt to convert numeric values back into numbers
    for k, v in header.metadata.items():
        try:
            header.metadata[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            # if it fails, leave it as a string
            pass

    return header


@to_ladybug.register(pd.Series)
def _(
    obj: pd.Series,
) -> Union[
    HourlyContinuousCollection,
    HourlyContinuousCollectionImmutable,
    HourlyDiscontinuousCollection,
    HourlyDiscontinuousCollectionImmutable,
    DailyCollection,
    DailyCollectionImmutable,
    MonthlyCollection,
    MonthlyCollectionImmutable,
    MonthlyPerHourCollection,
    MonthlyPerHourCollectionImmutable,
]:
    """Convert a pandas Series to a ladybug data collection."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")

    # create a new header from the series name
    header = to_ladybug(obj.name)

    # check for __type__ in header metadata
    if "__type__" not in header.metadata:
        raise ValueError(
            "Header metadata must contain a '__type__' key to determine the collection type."
        )

    collection = None
    match header.metadata["__type__"]:
        case "HourlyContinuousCollection" | "HourlyContinuousCollectionImmutable":
            analysis_period = to_ladybug(obj.index)
            header._analysis_period = analysis_period

            collection = HourlyContinuousCollection(
                header=header, values=obj.values.tolist()
            )
        case "HourlyDiscontinuousCollection" | "HourlyDiscontinuousCollectionImmutable":
            analysis_period = to_ladybug(obj.index)
            header._analysis_period = analysis_period

            datetimes = [
                DateTime(
                    month=dt.month,
                    day=dt.day,
                    hour=dt.hour,
                    minute=dt.minute,
                    leap_year=dt.is_leap_year,
                )
                for dt in obj.index
            ]
            collection = HourlyDiscontinuousCollection(
                header=header, values=obj.values.tolist(), datetimes=datetimes
            )
        case "DailyCollection" | "DailyCollectionImmutable":
            datetimes = [dt.dayofyear for dt in obj.index]
            collection = DailyCollection(
                header=header, values=obj.values.tolist(), datetimes=datetimes
            )
        case "MonthlyCollection" | "MonthlyCollectionImmutable":
            datetimes = [dt.month for dt in obj.index]
            collection = MonthlyCollection(
                header=header, values=obj.values.tolist(), datetimes=datetimes
            )
        case "MonthlyPerHourCollection" | "MonthlyPerHourCollectionImmutable":
            datetimes = [(dt.month, dt.hour, dt.minute) for dt in obj.index]
            collection = MonthlyPerHourCollection(
                header=header, values=obj.values.tolist(), datetimes=datetimes
            )
        case _:
            raise ValueError(f"Unknown collection type: {header.metadata['__type__']}")

    # remove the __type__ metadata key
    if "__type__" in collection.header.metadata:
        del collection.header.metadata["__type__"]

    return collection


@to_ladybug.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> EPW:
    """Convert a pandas DataFrame to ladybug EPW object."""

    # ensure that all columns are serialisable to ladybug collections
    collections: list[HourlyContinuousCollection] = []
    for _, series in obj.items():
        collections.append(to_ladybug(series))

    # check that time-zone is the same for all columns (and the index)
    time_zones = set()
    for collection in collections:
        if "time-zone" in collection.header.metadata:
            time_zones.add(collection.header.metadata["time-zone"])
    time_zones.add(obj.index[0].utcoffset().total_seconds() / 3600)
    if len(time_zones) > 1:
        raise ValueError(
            "All columns and the index must share the same time-zone metadata."
        )

    # determine whether leap year is needed
    is_leap_year = any(obj.index.is_leap_year)  # type: ignore

    # find any Ground Temperature columns and store in a dictionary ready to use later
    ground_temperatures: dict[float, MonthlyCollection] = {}
    for n, collection in enumerate(collections):
        if (
            "Ground Temperature" == collection.header.data_type.name
            and "depth" in collection.header.metadata
        ):
            # resample to monthly
            ground_temperatures[collection.header.metadata["depth"]] = (
                collection.average_monthly()
            )
            # remove this collection from the main list, and from the dataframe as well
            collections.pop(n)
            obj = obj.drop(columns=[obj.columns[n]])

    # create the EPW object
    epw = EPW.from_missing_values(is_leap_year=is_leap_year)

    # assign location metadata, using time-zone
    location = Location(
        longitude=np.interp(list(time_zones)[0], [-12, 0, 14], [-180, 0, 180]),  # type: ignore
        source="pandas DataFrame",
    )

    # assign collections to EPW object
    epw.location = location
    epw.monthly_ground_temperature = ground_temperatures
    for collection in collections:
        match collection.header.data_type.name:
            case "Dry Bulb Temperature":
                epw.dry_bulb_temperature.values = collection.to_unit("C").values
                epw.dry_bulb_temperature.header.metadata = collection.header.metadata
            case "Dew Point Temperature":
                epw.dew_point_temperature.values = collection.to_unit("C").values
                epw.dew_point_temperature.header.metadata = collection.header.metadata
            case "Relative Humidity":
                epw.relative_humidity.values = collection.to_unit("%").values
                epw.relative_humidity.header.metadata = collection.header.metadata
            case "Atmospheric Station Pressure":
                epw.atmospheric_station_pressure.values = collection.to_unit(
                    "Pa"
                ).values
                epw.atmospheric_station_pressure.header.metadata = (
                    collection.header.metadata
                )
            case "Direct Normal Radiation":
                epw.direct_normal_radiation.values = collection.to_unit("Wh/m2").values
                epw.direct_normal_radiation.header.metadata = collection.header.metadata
            case "Diffuse Horizontal Radiation":
                epw.diffuse_horizontal_radiation.values = collection.to_unit(
                    "Wh/m2"
                ).values
                epw.diffuse_horizontal_radiation.header.metadata = (
                    collection.header.metadata
                )
            case "Global Horizontal Radiation":
                epw.global_horizontal_radiation.values = collection.to_unit(
                    "Wh/m2"
                ).values
                epw.global_horizontal_radiation.header.metadata = (
                    collection.header.metadata
                )
            case "Wind Speed":
                epw.wind_speed.values = collection.to_unit("m/s").values
                epw.wind_speed.header.metadata = collection.header.metadata
            case "Wind Direction":
                epw.wind_direction.values = collection.to_unit("degrees").values
                epw.wind_direction.header.metadata = collection.header.metadata
            case "Total Sky Cover":
                epw.total_sky_cover.values = collection.to_unit("tenths").values
                epw.total_sky_cover.header.metadata = collection.header.metadata
            case "Opaque Sky Cover":
                epw.opaque_sky_cover.values = collection.to_unit("tenths").values
                epw.opaque_sky_cover.header.metadata = collection.header.metadata
            case "Visibility":
                epw.visibility.values = collection.to_unit("km").values
                epw.visibility.header.metadata = collection.header.metadata
            case "Ceiling Height":
                epw.ceiling_height.values = collection.to_unit("m").values
                epw.ceiling_height.header.metadata = collection.header.metadata
            case "Precipitable Water":
                epw.precipitable_water.values = collection.to_unit("mm").values
                epw.precipitable_water.header.metadata = collection.header.metadata
            case "Aerosol Optical Depth":
                epw.aerosol_optical_depth.values = collection.to_unit(
                    "thousandths"
                ).values
                epw.aerosol_optical_depth.header.metadata = collection.header.metadata
            case "Snow Depth":
                epw.snow_depth.values = collection.to_unit("cm").values
                epw.snow_depth.header.metadata = collection.header.metadata
            case "Liquid Precipitation Depth":
                epw.liquid_precipitation_depth.values = collection.to_unit("mm").values
                epw.liquid_precipitation_depth.header.metadata = (
                    collection.header.metadata
                )
            case "Liquid Precipitation Quantity":
                epw.liquid_precipitation_quantity.values = collection.to_unit(
                    "hr"
                ).values
                epw.liquid_precipitation_quantity.header.metadata = (
                    collection.header.metadata
                )
            case _:
                CONSOLE_LOGGER.warning(
                    f"Collection with data type '{collection.header.data_type.name}' not recognized in EPW. Did you pass the right kind of data (got {type(obj)})?"
                )

    return epw

@to_ladybug.register(pvlib_location)
def _(obj: pvlib_location) -> Location:
    """Convert a pvlib Location to a ladybug Location."""
    
    # set the datetime as base, to get the utc-offset
    year = AnalysisPeriod().datetimes[0].year
    dt = datetime(year, 1, 1)
    provided_offset_hours = ZoneInfo(obj.tz).utcoffset(dt).total_seconds() / 3600  # type: ignore
    likely_offset_hrs = get_tzinfo(latitude=obj.latitude, longitude=obj.longitude).utcoffset(
        dt
    ).total_seconds() / 3600  # type: ignore
    if not np.isclose((provided_offset_hours), likely_offset_hrs, atol=0.5):
        CONSOLE_LOGGER.warning(
            f"The provided timezone offset ({provided_offset_hours} hrs) does not match the expected offset based on latitude/longitude ({likely_offset_hrs} hrs). Using expected offset."
        )
        offset_hrs = likely_offset_hrs
    else:
        offset_hrs = provided_offset_hours
    
    likely_altitude = lookup_altitude(obj.latitude, obj.longitude)
    if not np.isclose(obj.altitude, likely_altitude, atol=50):
        CONSOLE_LOGGER.warning(
            f"The provided altitude ({obj.altitude} m) does not match the expected altitude based on latitude/longitude ({likely_altitude} m). Using expected altitude."
        )
        altitude = likely_altitude
    else:
        altitude = obj.altitude
    
    # ladybug location time_zone must be between -12 and 14
    if offset_hrs > 14:
        offset_hrs = offset_hrs - 24
    elif offset_hrs < -12:
        offset_hrs = offset_hrs + 24
    
    return Location(
        latitude=obj.latitude,
        longitude=obj.longitude,
        elevation=altitude, # type: ignore
        time_zone=offset_hrs,
        source=obj.name,
    )
