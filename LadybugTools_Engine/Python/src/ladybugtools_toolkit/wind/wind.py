import calendar
import copy
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from honeybee.shade import Shade
from ladybug.dt import Date
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.sunpath import Location
from ladybug.windrose import WindRose #TODO
from ladybug_geometry.geometry3d import Point3D, Ray3D, Vector3D
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

from ..convert.to_colour import to_colour
from ..convert.to_ladybug import to_ladybug
from ..convert.to_pandas import to_pandas
from ..honeybee_energy_extension.util import get_schedule_as_data_collection #TODO
from ..ladybug_extension.analysisperiod import DefaultAnalysisPeriod

from ..ladybug_extension.location import (
    average_location,
    location_to_pytz_fixed_offset,
    location_to_timezone,
)

from ..ladybug_extension.analysisperiod import (
    analysis_period_to_string,
    metadata_dict_to_str, #TODO
    metadata_str_to_dict, #TODO
)

from ..helpers import(
    cardinality,
    angle_to_vector as azimuth_to_vector,
    circular_weighted_mean,
)

from ..ladybug_geometry_extension.util import (
    vector_to_azimuth_altitude,
)

from ..scrape_weather import OpenMeteoVariable, openmeteo #TODO
from ..plot.utilities import contrasting_color
from .terrain import WindTerrainType
from .util import direction_bin_edges

@dataclass
class Wind:
    """An object containing wind data.

    Args:
        location (Location): A ladybug Location object.
        datetimes (pd.DatetimeIndex): An iterable of datetime-like objects.
        wind_speed (list[float]): Wind speeds in m/s.
        wind_direction (list[float]): Wind directions in degrees clockwise from
            north (0).
        height_above_ground (float): Height above ground in meters where wind
            data was collected.
        terrain_type (TerrainType): The terrain type associated with
            this wind data. Defaults to TerrainType.COUNTRY.

    Raises:
        ValueError: If input data validation fails.

    """

    # NOTE - BE STRICT WITH THE TYPING!
    # NOTE - Conversions happen in class methods.
    # NOTE - Validation happens at instantiation.

    location: Location #TODO: change reference of self.location.source to self.source
    datetimes: pd.DatetimeIndex
    wind_speed: list[float]
    wind_direction: list[float]
    height_above_ground: float
    terrain_type: WindTerrainType = WindTerrainType.COUNTRY
    source: str

    # region: DUNDER METHODS

    def __post_init__(self):
        """Check for validation of the inputs."""
        # location checks
        if not isinstance(self.location, Location):
            raise TypeError("location must be a ladybug Location object.")
        if self.source is None:
            warnings.warn(
                'The source input is None. This means that things are a bit ambiguous! A default value of "UnknownSource" has been added.'
            )
            self.source = "UnknownSource"

        # datetimes validation
        self.datetimes = pd.DatetimeIndex(self.datetimes, freq="infer")
        # check datetimes are increasing, and unique, and don't contain nulls
        if any(self.datetimes.duplicated()):
            raise ValueError("datetimes must be unique.")
        if any(self.datetimes.isna()):
            raise ValueError("datetimes cannot contain null values.")
        if any(np.diff(self.datetimes) < timedelta(0)):
            raise ValueError("datetimes must be in increasing order.")

        # timezone validation
        if self.datetimes[0].tzinfo is None:
            self.datetimes = self.datetimes.tz_localize(
                location_to_timezone(self.location), ambiguous="NaT"
            )

        # height above ground validation
        if not isinstance(self.height_above_ground, (int, float)):
            raise TypeError("height_above_ground must be a number.")
        if self.height_above_ground < 0.1:
            raise ValueError(
                "height_above_ground must be greater than or equal to 0.1."
            )

        # terrain type validation
        if self.terrain_type is None:
            self.terrain_type = WindTerrainType.COUNTRY
            warnings.warn(
                "terrain_type was not provided. Defaulting to TerrainType.COUNTRY."
            )
        if not isinstance(self.terrain_type, WindTerrainType):
            raise TypeError("terrain_type must be a TerrainType object.")

        # data validation
        array_names = [
            "wind_speed",
            "wind_direction",
        ]
        for name in array_names:
            if len(getattr(self, name)) != len(self.datetimes):
                raise ValueError(
                    f"{name} must be the same length as datetimes. {len(getattr(self, name))} != {len(self.datetimes)}."
                )
            if not all(
                isinstance(i, (int, float, np.float64)) for i in getattr(self, name)
            ):
                raise TypeError(f"{name} must be a list of numeric values.")
            if any(np.isnan(getattr(self, name))):
                raise ValueError(f"{name} cannot contain null values.")
        if any(i < 0 for i in self.wind_speed):
            raise ValueError("Wind speeds cannot be negative.")
        if any(i < 0 or i > 360 for i in self.wind_direction):
            raise ValueError(
                f"Wind directions must be between 0 and 360 degrees, values given span {min(self.wind_direction)} to {max(self.wind_direction)}."
            )

    def __len__(self) -> int:
        return len(self.datetimes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} data from {self.location.source}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.location,
                tuple(self.datetimes),
                tuple(self.wind_speed),
                tuple(self.wind_direction),
                str(self.terrain_type),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wind):
            return False
        return (
            self.location == other.location
            and all(self.datetimes == other.datetimes)
            and self.wind_speed == other.wind_speed
            and self.wind_direction == other.wind_direction
            and self.terrain_type == other.terrain_type
        )

    def __iter__(self) -> Generator[tuple[pd.Timestamp, float, float]]:
        for i in range(len(self)):
            yield (self.datetimes[i], self.wind_speed[i], self.wind_direction[i])

    def __getitem__(self, idx: int) -> dict[str, Union[datetime, float]]:
        return {
            "datetime": self.datetimes[idx],
            "wind_speed": self.wind_speed[idx],
            "wind_direction": self.wind_direction[idx],
        }

    def __copy__(self) -> "Wind":
        """Return a shallow copy of the Wind object."""
        return Wind(
            location=self.location.duplicate(),
            datetimes=self.datetimes.copy(),
            wind_speed=copy.copy(self.wind_speed),
            wind_direction=copy.copy(self.wind_direction),
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    # endregion: DUNDER METHODS

    # region: PROPERTIES

    @property
    def source(self) -> str:
        """Return the source for this object."""
        return self.location.source

    @property
    def _metadata_str(self) -> str:
        """Return the metadata for this object as a string."""
        return metadata_dict_to_str({"source": self.location.source, "location": str(self.location), "time-zone": self.location.time_zone, "terrain_type": self.terrain_type.name, "height_above_ground": self.height_above_ground})


    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata for this object as a dictionary."""
        d = self.location.to_dict()
        d["terrain_type"] = self.terrain_type.name
        d["height_above_ground"] = self.height_above_ground
        d["period"] = (
            min(self.datetimes).isoformat() + " to " + max(self.datetimes).isoformat()
        )
        # drop the "type" key
        del d["type"]
        return d

    @property
    def start_datetime(self) -> date:
        return min(self.datetimes)

    @property
    def end_datetime(self) -> date:
        return max(self.datetimes)

    @property
    def lb_datetimes(self) -> list[date]:
        return [to_ladybug(dt) for dt in self.datetimes]

    @property
    def lb_dates(self) -> list[Date]:
        return [to_ladybug(d) for d in self.datetimes]

    @property
    def wind_speed_series(self) -> pd.Series:
        return pd.Series(
            data=self.wind_speed,
            index=self.datetimes,
            name=("Wind Speed", "m/s", metadata_dict_to_str(self.metadata)),
        )

    @property
    def wind_speed_collection(self) -> HourlyContinuousCollection:
        # create an aggregate dataset
        series = self.wind_speed_series
        agg_year = series.groupby(
            [series.index.month, series.index.day, series.index.hour]
        ).mean()
        if (2, 29) in agg_year.index:
            # if leap day is present, we need to remove it
            agg_year = agg_year.drop((2, 29))
        # add a generic index
        agg_year.index = to_pandas(AnalysisPeriod())
        agg_year.name = (
            agg_year.name[0],
            agg_year.name[1],
            agg_year.name[2] + " | __type__: HourlyContinuousCollection",
        )

        return to_ladybug(agg_year)

    @property
    def wind_direction_series(self) -> pd.Series:
        return pd.Series(
            data=self.wind_direction,
            index=self.datetimes,
            name=("Wind Direction", "degrees", metadata_dict_to_str(self.metadata)),
        )

    @property
    def wind_direction_collection(self) -> HourlyContinuousCollection:
        # create an aggregate dataset
        series = self.wind_direction_series
        agg_year = (
            series.groupby(
                [series.index.month, series.index.day, series.index.hour]
            ).apply(circular_weighted_mean)
            % 360
        )
        if (2, 29) in agg_year.index:
            # if leap day is present, we need to remove it
            agg_year = agg_year.drop((2, 29))
        # add a generic index
        agg_year.index = to_pandas(AnalysisPeriod())
        agg_year.name = (
            agg_year.name[0],
            agg_year.name[1],
            agg_year.name[2] + " | __type__: HourlyContinuousCollection",
        )

        return to_ladybug(agg_year)

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.wind_speed_series,
                self.wind_direction_series,
            ],
            axis=1,
        )

    @property
    def uv(self) -> pd.DataFrame:
        """Return the U and V wind components in m/s as a pd.DataFrame."""
        u, v = azimuth_to_vector(self.wind_direction)
        return pd.concat([u * self.wind_speed_series, v * self.wind_speed_series], axis=1, keys=["u", "v"])

    # endregion: PROPERTIES

    # region: CLASS METHODS

    @classmethod
    def from_epw(
        cls, epw: Union[Path, str, EPW], terrain_type: WindTerrainType = WindTerrainType.COUNTRY
    ) -> "Wind":
        """Create a Wind object from an EPW file or object.

        Args:
            epw (Union[Path, str, EPW]):
                The path to the EPW file, or an EPW object.

        """
        if isinstance(epw, (str, Path)):
            epw = EPW(epw)

        location = epw.location

        # obtain the datetimes
        datetimes = to_pandas(epw.dry_bulb_temperature.header.analysis_period)
        datetimes = datetimes.tz_localize(
            location_to_pytz_fixed_offset(location)
        )

        return cls(
            location=location,
            datetimes=datetimes,
            wind_speed=epw.wind_speed.values,
            wind_direction=epw.wind_direction.values,
            height_above_ground=10,
            terrain_type=terrain_type,
            source=f"{Path(epw.file_path).name}"
        )

    def to_dict(self) -> dict:
        """Represent the object as a python-native dtype dictionary."""
        return {
            "type": "Wind",
            "location": self.location.to_dict(),
            "datetimes": [i.isoformat() for i in self.datetimes],
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "height_above_ground": self.height_above_ground,
            "terrain_type": self.terrain_type.name,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Wind":
        """Create this object from a dictionary."""
        if d.get("type", None) != "Wind":
            raise ValueError("The dictionary cannot be converted Wind object.")

        return cls(
            location=Location.from_dict(d["location"]),
            datetimes=pd.to_datetime(d["datetimes"]),
            wind_speed=d["wind_speed"],
            wind_direction=d["wind_direction"],
            height_above_ground=d["height_above_ground"],
            terrain_type=WindTerrainType[d["terrain_type"]],  # type: ignore[call-arg]
            source = d.pop(["source"], "Unknown Python Dict"),
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Wind":
        """Create this object from a JSON string."""
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        wind_speed_column: str,
        wind_direction_column: str,
        location: Optional[Location] = None,
        terrain_type: Optional[WindTerrainType] = None,
        height_above_ground: Optional[float] = None,
        source: str = "DataFrame",
    ) -> "Wind":
        """Create this object from a DataFrame.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing the wind data.
            location (Location, optional):
                A ladybug Location object. If not provided, the location data
                will be extracted from the DataFrame if present.
            terrain_type (TerrainType, optional):
                The terrain type associated with the wind data. If not provided,
                the default is TerrainType.COUNTRY, or the terrain type from the
                dataframe metadata if present.
            height_above_ground (float, optional):
                The height above ground (in m) where the input wind speeds and
                directions were collected. If not provided, the default is 10m,
                or the height from the dataframe metadata if present.

        """
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be of type {pd.DataFrame}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("The DataFrame's index must be of type pd.DatetimeIndex.")

        if not isinstance(location, Location):
            raise TypeError("location must be a ladybug Location object.")

        # remove NaN values
        df.dropna(axis=0, how="any", inplace=True)

        # remove duplicates in input dataframe
        df = df.loc[~df.index.duplicated()]

        ws_series:pd.Series = df[wind_speed_column]
        wd_series:pd.Series = df[wind_direction_column]

        loc_copy = location.duplicate()

        return cls(
            location=loc_copy,
            datetimes=df.index,
            wind_speed=ws_series.values,
            wind_direction=wd_series.values,
            height_above_ground=height_above_ground,
            terrain_type=terrain_type,
            source = source,
        )

    @classmethod
    def from_average(cls, objects: list["Wind"], weights: list[float] = None) -> "Wind":
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""
        # validation
        if not all(isinstance(i, Wind) for i in objects):
            raise TypeError("objects must be a list of Wind objects.")
        if len(objects) == 0:
            raise ValueError("objects cannot be empty.")
        if len(objects) == 1:
            return objects[0]

        # check datetimes are the same
        for obj in objects:
            if not all(obj.datetimes == objects[0].datetimes):
                raise ValueError("All objects must share the same datetimes.")

        # create default weightings if None
        if weights is None:
            weights = [1 / len(objects)] * len(objects)
        else:
            if sum(weights) != 1:
                raise ValueError("weights must total 1.")

        # create average location
        avg_location = average_location([i.location for i in objects], weights=weights)
        
        source = "|".join(
            [
                str(w.source) if w.source not in ["", "-", None] else "NoSource"
                for w in objects
            ]
        )

        # align collections so that intersection only is created
        df_ws = pd.concat([i.wind_speed_series for i in objects], axis=1).dropna()
        df_wd = pd.concat([i.wind_direction_series for i in objects], axis=1).dropna()

        # construct the weighted means
        wd_avg = np.array(
            [circular_weighted_mean(i, weights) for _, i in df_wd.iterrows()]
        )
        ws_avg = np.average(df_ws, axis=1, weights=weights)

        # construct the avg height above ground
        avg_height_above_ground = np.average(
            [i.height_above_ground for i in objects], weights=weights
        )

        # construct the new terrain type, based on the average of the input objects
        avg_roughness_length = np.average(
            [i.terrain_type.roughness_length for i in objects], weights=weights
        )
        terrain_type = WindTerrainType.from_roughness_length(avg_roughness_length)

        # return the new averaged object
        return cls(
            wind_speed=ws_avg.tolist(),
            wind_direction=wd_avg.tolist(),
            datetimes=objects[0].datetimes,
            height_above_ground=avg_height_above_ground,
            location=avg_location,
            terrain_type=terrain_type,
            source=source
        )

    @classmethod
    def from_uv(
        cls,
        u: list[float],
        v: list[float],
        location: Location,
        datetimes: list[datetime],
        height_above_ground: float = 10,
        terrain_type: WindTerrainType = WindTerrainType.COUNTRY,
        source: str = "Custom U, V wind components",
    ) -> "Wind":
        """Create a Wind object from a set of U, V wind components.

        Args:
            u (list[float]):
                An iterable of U (eastward) wind components in m/s.
            v (list[float]):
                An iterable of V (northward) wind components in m/s.
            datetimes (list[datetime]):
                An iterable of datetime-like objects.
            height_above_ground (float, optional):
                The height above ground (in m) where the input wind speeds and
                directions were collected.
                Defaults to 10m.
            source (str, optional):
                A source string to describe where the input data comes from.
                Defaults to None.

        Returns:
            Wind:
                A Wind object!

        """
        # convert UV into angle and magnitude
        wind_direction = []
        for uu, vv in zip(u, v):
            wind_direction.append(vector_to_azimuth_altitude(Vector3D(uu, vv, 0)[0]))
        wind_speed = np.sqrt(np.square(u) + np.square(v))

        if any(wind_direction[wind_speed == 0] == 90):
            warning_message = "Some input vectors have velocity of 0. This is not bad, but can mean directions may be misreported."
            warnings.warn(warning_message, UserWarning)

        return cls(
            wind_speed=wind_speed.tolist(),
            wind_direction=wind_direction.tolist(),
            datetimes=datetimes,
            height_above_ground=height_above_ground,
            location=location,
            terrain_type=terrain_type,
            source=source,
        )

    @classmethod
    def from_openmeteo(
        cls,
        location: Location,
        start_date: Union[str, date],
        end_date: Union[str, date],
        terrain_type: WindTerrainType = WindTerrainType.COUNTRY,
    ) -> "Wind":
        """Query Openmeteo for wind data."""
        df = openmeteo(
            location=location,
            start_date=start_date,
            end_date=end_date,
            variables=(
                OpenMeteoVariable.WIND_SPEED_10M,
                OpenMeteoVariable.WIND_DIRECTION_10M,
            ),
        )
        datetimes: pd.DatetimeIndex = df.index
        ws: pd.Series = df["Wind Speed"].squeeze()  # type: ignore
        wd: pd.Series = df["Wind Direction"].squeeze()  # type: ignore

        # modify location to state the Openmeteo file in the source field
        loc = location.duplicate()

        return cls(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws.values.tolist(),
            wind_direction=wd.values.tolist(),
            height_above_ground=10,
            terrain_type=terrain_type,
            source=f"{metadata_str_to_dict(wd.name[1])['source']} [{datetimes.min():%Y-%m-%d}-{datetimes.max():%Y-%m-%d}, n={len(ws):,}]"
        )

    # endregion: CLASS METHODS

    # region: FILTER METHODS

    def filter_by_boolean_mask(self, mask: Optional[list[bool]] = None) -> "Wind":
        """Filter the current object by a boolean mask.

        Args:
            mask (list[bool]):
                A boolean mask to filter the current object.

        Returns:
            Wind:
                A dataset describing solar radiation.

        """
        if mask is None:
            mask = [True] * len(self)

        # validations
        if not all(isinstance(i, bool) for i in mask):
            raise TypeError("mask must be a list of booleans.")
        if len(mask) != len(self):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )
        if sum(mask) == 0:
            raise ValueError("No data remains within the given boolean filters.")
        if sum(mask) == len(self):
            return self

        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered)"

        return Wind(
            location=loc,
            datetimes=[i for i, j in zip(*[self.datetimes, mask]) if j],
            wind_speed=[i for i, j in zip(*[self.wind_speed, mask]) if j],
            wind_direction=[i for i, j in zip(*[self.wind_direction, mask]) if j],
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> "Wind":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Wind:
                A dataset describing wind.

        """
        mask = []
        for n, i in enumerate(self.lb_datetimes):
            mask.append(i in analysis_period.datetimes)

        # create new data
        loc = self.location.duplicate()
        loc.source = (
            f"{self.location.source} (filtered to {analysis_period_to_string(analysis_period)})",
        )
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_direction=wd,
            wind_speed=ws,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_time(
        self,
        years: Optional[list[int]] = None,
        months: Optional[list[float]] = None,
        days: Optional[list[float]] = None,
        hours: Optional[list[int]] = None,
    ) -> "Wind":
        """Filter the current object by months, days, hours.

        Args:
            years (list[int], optional):
                A list of years to include.
                Default to all years.
            months (list[int], optional):
                A list of months.
                Defaults to all possible months.
            days (list[int], optional):
                A list of days.
                Defaults to all possible days.
            hours (list[int], optional):
                A list of hours.
                Defaults to all possible hours.

        Returns:
            Wind:
                A dataset describing historic wind data.

        """
        idx = self.datetimes
        filtered_by = []
        if years is None:
            years = idx.year.unique().tolist()
        else:
            filtered_by.append("year")
        if months is None:
            months = list(range(1, 13))
        else:
            filtered_by.append("month")
        if days is None:
            days = list(range(1, 32))
        else:
            filtered_by.append("day")
        if hours is None:
            hours = list(range(0, 24))
        else:
            filtered_by.append("hour")

        if len(filtered_by) > 2:
            filtered_by = ", ".join(filtered_by[:-1]) + ", and " + str(filtered_by[-1])
        elif len(filtered_by) == 2:
            filtered_by = " and ".join(filtered_by)
        elif len(filtered_by) == 1:
            filtered_by = filtered_by[0]

        # construct masks
        year_mask = idx.year.isin(years)
        month_mask = idx.month.isin(months)
        day_mask = idx.day.isin(days)
        hour_mask = idx.hour.isin(hours)
        mask = np.all([year_mask, month_mask, day_mask, hour_mask], axis=0)

        # create new data
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered by {filtered_by})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_direction(
        self,
        left_angle: float = 0,
        right_angle: float = 360,
        include_left: bool = True,
        include_right: bool = True,
    ) -> "Wind":
        """Filter the current object by wind direction, based on the angle as
        observed from a location.

        Args:
            left_angle (float):
                The left-most angle, to the left of which wind speeds and
                directions will be removed.
                Defaults to 0.
            right_angle (float):
                The right-most angle, to the right of which wind speeds and
                directions will be removed.
                Defaults to 360.
            include_left (bool, optional):
                Include values that are exactly the left angle.
                Defaults to True.
            include_right (bool, optional):
                Include values that are exactly the right angle.
                Defaults to True.

        Return:
            Wind:
                A Wind object!

        """
        if left_angle < 0 or right_angle > 360:
            raise ValueError("Angle limits must be between 0 and 360 degrees.")

        if left_angle == 0 and right_angle == 360:
            return self

        if (left_angle == right_angle) or (left_angle == 360 and right_angle == 0):
            raise ValueError("Angle limits cannot be identical.")

        wd = self.wind_direction_series.values

        if include_right:
            right_mask = wd <= right_angle
        else:
            right_mask = wd < right_angle

        if include_left:
            left_mask = wd >= left_angle
        else:
            left_mask = wd > left_angle

        if left_angle > right_angle:
            mask = left_mask | right_mask
        else:
            mask = left_mask & right_mask

        # create new data
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered by direction {'[' if include_left else '('}{left_angle}°-{right_angle}°{']' if include_right else ')'})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_speed(
        self,
        min_speed: float = 0,
        max_speed: float = np.inf,
        include_left: bool = True,
        include_right: bool = True,
    ) -> "Wind":
        """Filter the current object by wind speed, based on given low-high limit values.

        Args:
            min_speed (float):
                The lowest speed to include. Values below this wil be removed.
                Defaults to 0.
            max_speed (float):
                The highest speed to include. Values above this wil be removed.
                Defaults to np.inf.
            include_right (bool, optional):
                Include values that are exactly the max speed.
                Defaults to True.
            include_left (bool, optional):
                Include values that are exactly the min speed.

        Return:
            Wind:
                A Wind object!

        """
        if min_speed < 0:
            raise ValueError("min_speed cannot be negative.")

        if max_speed <= min_speed:
            raise ValueError("min_speed must be less than max_speed.")

        if min_speed == 0 and np.isinf(max_speed):
            return self

        ws = self.wind_speed_series.values
        if include_right:
            right_mask = ws <= max_speed
        else:
            right_mask = ws < max_speed

        if include_left:
            left_mask = ws >= min_speed
        else:
            left_mask = ws > min_speed

        mask = left_mask & right_mask

        # create new data
        loc = self.location.duplicate()
        speed_range = f"{'[' if include_left else '('}{min_speed}m/s-{max_speed}m/s{']' if include_right else ')'}"
        loc.source = f"{self.location.source} (filtered by speed {speed_range})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    # endregion: FILTER METHODS

    # region: INSTANCE METHODS

    def _direction_categories(self, directions: int = 36) -> pd.Categorical:
        edges = direction_bin_edges(directions=directions)
        return pd.cut(self.wind_direction, bins=edges, include_lowest=True, right=True)

    def _direction_binned_data(
        self, directions: int = 36, other_data: Any = None
    ) -> dict[str, list[Any]]:
        """Bin data by wind direction."""
        if other_data is None:
            other_data = self.wind_speed
        if len(other_data) != len(self):
            raise ValueError("other_data must be same length as this object")

        binned = self._direction_categories(directions=directions)
        grp = pd.Series(other_data).groupby(binned, observed=True)
        d = {k: table.values.tolist() for k, table in grp}
        # combine the first and last bins
        renamer = {}
        for n, interval in enumerate(binned.categories):
            if n == 0 or n == len(binned.categories) - 1:
                renamer[interval] = (
                    (
                        str(binned.categories[-1]).split(",")[0]
                        if directions != 1
                        else "(0.0"
                    )
                    + ","
                    + str(binned.categories[0]).split(",")[1]
                )
            else:
                renamer[interval] = str(interval)
        # rename the keys in the original dict
        d_renamed = {}
        for k, v in d.items():
            target_key = renamer[k]
            if target_key in d_renamed:
                d_renamed[target_key].extend(v)
            else:
                d_renamed[target_key] = v
        return d_renamed

    def proportion_calm(self, threshold: float = 0.1) -> float:
        """Return proportion of timestep's below calm threshold.

        Args:
            threshold (float, optional): Threshold for calm wind speeds.
                Defaults to 0.1.

        Returns:
            float: Proportion of calm instances.

        """
        s = self.wind_speed_series
        return float((s <= threshold).sum() / len(s))

    def calm_mask(self, threshold: float = 0.1) -> list[bool]:
        """Return boolean mask of timestep's below calm threshold.

        Args:
            threshold (float, optional): Threshold for calm wind speeds.
                Defaults to 0.1.

        Returns:
            list[bool]: Boolean mask of calm timestep's.

        """
        return (np.array(self.wind_speed) <= threshold).tolist()

    def percentile(
        self,
        q: Union[float, Tuple[float, ...]] = (0.25, 0.5, 0.75, 0.95),
        directions: int = 8,
    ) -> pd.DataFrame:
        """Calculate wind speed at given percentiles.

        Args:
            q (Union[float, Tuple[float, ...]], optional): Percentiles to
                calculate. Defaults to (0.25, 0.5, 0.75, 0.95).
            directions (int, optional): Number of direction bins.
                Defaults to 8.

        Returns:
            pd.DataFrame: Wind speeds at specified percentiles by direction.

        """
        q = np.atleast_1d(q)
        dd = self._direction_binned_data(directions=directions)
        return pd.DataFrame(
            {k: np.quantile(v, q).tolist() for k, v in dd.items()}, index=q
        ).T

    def to_height(
        self,
        target_height: float,
        log_law: bool = True,
    ) -> "Wind":
        """Translate wind data to a different height above ground.

        Args:
            target_height (float): Height to translate to in meters.
            log_law (bool, optional): Whether to use log or power function.
                Defaults to True.

        Returns:
            Wind: Translated Wind object.

        """
        if self.height_above_ground == target_height:
            return self

        wss = [
            self.terrain_type.wind_speed_at_height(
                reference_value=ws,
                reference_height=self.height_above_ground,
                target_height=target_height,
                log_law=log_law,
            )
            for ws in self.wind_speed
        ]
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} translated to {target_height}m"
        return Wind(
            wind_speed=wss,
            wind_direction=self.wind_direction,
            datetimes=self.datetimes,
            height_above_ground=target_height,
            location=loc,
            terrain_type=self.terrain_type,
        )

    def apply_directional_factors(
        self, directions: int, factors: tuple[float]
    ) -> "Wind":
        """Adjust wind speeds by directional factors.

        Factors start at north and move clockwise. Right edges are inclusive.

        Example:
            >>> wind = Wind.from_epw(epw_path)
            >>> wind.apply_directional_factors(
            ...     directions=4,
            ...     factors=(0.5, 0.75, 1, 0.75)
            ... )

        Args:
            directions (int): Number of direction bins.
            factors (tuple[float]): Adjustment factors per direction bin.

        Returns:
            Wind: Adjusted Wind object.

        Raises:
            ValueError: If number of factors doesn't match directions.

        """
        binned = self._direction_categories(directions=directions)

        if len(binned.categories) - 1 != len(factors):
            raise ValueError("Number of factors must be equal to number of directions.")

        mapping = {k: v for k, v in zip(binned.categories, factors + [factors[0]])}

        wind_speeds = self.wind_speed * binned.map(mapping, na_action="ignore")

        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (adjusted by {directions} directional factors {factors})"

        return Wind(
            wind_speed=wind_speeds.tolist(),
            wind_direction=self.wind_direction,
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
            location=loc,
            terrain_type=self.terrain_type,
        )

    def direction_counts(
        self,
        directions: int = 8,
        as_midpoints: bool = False,
    ) -> dict[str, int]:
        """Calculate number of values per wind direction.

        Args:
            directions (int, optional): Number of direction bins.
                Defaults to 8.
            as_midpoints (bool, optional): Return midpoint angles instead of
                bin ranges. Defaults to False.

        Returns:
            dict[str, int]: Count of values per direction bin.

        """
        dd = {
            k: len(np.array(v)[np.array(v) != 0])
            for k, v in self._direction_binned_data(directions=directions).items()
        }
        if as_midpoints:
            lookup = {}
            for n, (k, v) in enumerate(dd.items()):
                if n == 0:
                    lookup[k] = 0.0
                else:
                    lookup[k] = (
                        float(k[1:-1].split(",")[0]) + float(k[1:-1].split(",")[1])
                    ) / 2.0
            dd = {lookup[k]: v for k, v in dd.items()}
        return dd

    def prevailing(
        self, directions: int = 8, n: int = 1, as_cardinal: bool = False
    ) -> list[str]:
        """Get prevailing wind directions.

        Args:
            directions (int, optional): Number of direction bins.
                Defaults to 8.
            n (int, optional): Number of directions to return.
                Defaults to 1.
            as_cardinal (bool, optional): Return cardinal directions.
                Defaults to False.

        Returns:
            list[str]: Prevailing wind directions.

        """
        pp = self.direction_counts(directions=directions, as_midpoints=True)
        prevailing_directions = [
            i[0] for i in sorted(pp.items(), key=lambda x: x[1], reverse=True)
        ]
        if as_cardinal:
            x = [cardinality(j, directions=32) for j in prevailing_directions]
            # remove duplicates from x, but retain order
            seen = []
            for i in x:
                if i not in seen:
                    seen.append(i)
                if len(seen) == n:
                    break
            return seen
        return prevailing_directions[:n]

    def month_hour_mean_matrix(
        self, other_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculate mean wind data for each month and hour.

        Args:
            other_data (HourlyContinuousCollection, optional): Additional data
                to include in matrix. Defaults to wind speed.

        Returns:
            pd.DataFrame: Matrix of averaged values by month and hour.

        Raises:
            ValueError: If other_data has invalid format or length.

        """
        # ensure data is suitable for matricisation
        if other_data is None:
            other_data = self.wind_speed_series

        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as the wind data.")
        # if not isinstance(other_data, HourlyContinuousCollection):
        #     raise ValueError("other_data must be a HourlyContinuousCollection.")

        # convert other data to a series
        if isinstance(other_data, pd.Series):
            other_data_series = other_data
        else:
            other_data_series = to_pandas(other_data)

        # get the average wind direction per-hour, per-month
        wd = self.wind_direction_series
        wind_directions = (
            (
                (
                    wd.groupby(
                        [self.datetimes.month, self.datetimes.hour], axis=0
                    ).apply(circular_weighted_mean)
                )
                % 360
            )
            .unstack()
            .T
        )

        _other_data = (
            other_data_series.groupby(
                [self.datetimes.month, self.datetimes.hour], axis=0
            )
            .mean()
            .unstack()
            .T
        )

        df = pd.concat(
            [wind_directions, _other_data],
            axis=1,
            keys=[wd.name, other_data_series.name],
        )
        df.index.name = "hour"
        df.columns.set_names(names=["variable", "month"], level=[0, 1], inplace=True)

        return df

    def windrose(
        self,
        other_data: Optional[HourlyContinuousCollection] = None,
        directions: int = 36,
    ) -> WindRose:
        """Create a WindRose object.

        Args:
            other_data (HourlyContinuousCollection, optional): Additional data
                to include. Defaults to wind speed.
            directions (int, optional): Number of direction bins.
                Defaults to 36.

        Returns:
            WindRose: A WindRose visualization object.

        """
        if other_data is None:
            other_data = self.wind_speed_collection
        return WindRose(
            direction_data_collection=self.wind_direction_collection,
            analysis_data_collection=other_data,
            direction_count=directions,
        )

    def histogram(
        self,
        directions: int = 36,
        other_data: Optional[list[float]] = None,
        other_bins: Union[list[float], int, None] = 11,
        density: bool = False,
    ) -> pd.DataFrame:
        """Bin data by direction and return counts.

        Args:
            directions (int, optional): Number of direction bins.
                Defaults to 36.
            other_data (list[float], optional): Additional data to bin.
                Defaults to wind speed.
            other_bins (Union[list[float], int, None], optional): Bins for
                other data. Defaults to 11.
            density (bool, optional): Return probability density.
                Defaults to False.

        Returns:
            pd.DataFrame: Binned data counts/probabilities.

        Raises:
            ValueError: If bin edges are invalid.

        """
        # get other data
        if other_data is None:
            other_data = self.wind_speed
        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as wind data")

        # bin per direction
        dd = self._direction_binned_data(directions=directions)

        # create other intervals, and check for invalid edges
        cats = pd.cut(other_data, other_bins, right=True, include_lowest=True)
        # if cats.categories[-1].right < max(other_data) or cats.categories[0].left > min(
        #     other_data
        # ):
        #     raise ValueError(
        #         f"bin edges must be between {min(other_data)} and {max(other_data)} (inclusive)"
        #     )

        # iterate binned data, and bin
        new_d = {}
        for k, v in dd.items():
            dv = (
                pd.cut(v, other_bins, right=True, include_lowest=True)
                .value_counts()
                .values
            )
            new_d[k] = {str(i): float(j) for i, j in zip(*[cats.categories, dv])}

        df = pd.DataFrame(new_d).T

        # rename first column
        if float(str(df.columns[0]).split(",")[0][1:]) < min(other_data):
            r = str(df.columns[0]).split(",")[1]
            df.rename(columns={df.columns[0]: f"({min(other_data)},{r}"}, inplace=True)

        # name the index and columns to be used downstream
        df.index.name = "Wind Direction (degrees)"

        if density:
            return df / df.values.sum()

        return df

    # endregion: INSTANCE METHODS

    # region: visualization

    def plot_windprofile(
        self,
        ax: Axes = None,
        max_height: int = 30,
        log_law: bool = True,
        terrain_types: tuple[WindTerrainType] = None,
    ) -> Axes:
        reference_value = float(np.mean(self.wind_speed))

        if terrain_types is None:
            terrain_types = tuple([i for i in WindTerrainType])
        if not all(isinstance(tt, WindTerrainType) for tt in terrain_types):
            raise TypeError("terrain_types must be a list of TerrainType objects.")

        if ax is None:
            ax = plt.gca()

        heights = np.arange(0, max_height, 1)
        speeds = []
        for target_terrain in terrain_types:
            speeds.append(
                [
                    self.terrain_type.wind_speed_at_height(
                        reference_value=reference_value,
                        reference_height=self.height_above_ground,
                        target_height=height,
                        log_law=log_law,
                        target_terrain_type=target_terrain,
                    )
                    for height in heights
                ]
            )

        # add the reference value to the plot
        ax.scatter(reference_value, self.height_above_ground, c="k")
        ax.plot(
            [reference_value] * 2,
            [0, self.height_above_ground],
            c="k",
            alpha=0.5,
            lw=2,
            ls="--",
        )
        ax.plot(
            [0, reference_value],
            [self.height_above_ground, self.height_above_ground],
            c="k",
            lw=2,
            ls="--",
            alpha=0.5,
        )
        ax.text(
            reference_value + 0.02,
            0.1,
            f"{reference_value:0.2f} m/s",
            ha="left",
            va="bottom",
        )

        for speed, tt in zip(*[speeds, terrain_types]):
            ax.plot(speed, heights, lw=2, label=tt.name)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Height (m)")
        ax.set_ylim(0, max_height)
        ax.set_xlim(0, np.array(speeds).max() + 0.1)
        ax.set_title(
            f"Wind Profiles (using {'log' if log_law else 'power'}-law)\n{self}"
        )
        ax.legend()

        return ax

    def plot_windmatrix(
        self,
        ax: Optional[Axes] = None,
        show_values: bool = True,
        show_arrows: bool = True,
        other_data: Optional[Union[HourlyContinuousCollection, pd.Series]] = None,
        **kwargs,
    ) -> Axes:
        """Create a plot showing the annual wind speed and direction bins
        using the month_time_average method.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. If None, the current axes will be used.
            show_values (bool, optional):
                Whether to show values in the cells.
                Defaults to True.
            show_arrows (bool, optional):
                Whether to show the directional arrows on each patch.
                Defaults to True.
            other_data: (pd.Series, optional):
                The other data to align with the wind direction and speed.
                Defaults to None which uses wind speed.
            **kwargs:
                Additional keyword arguments to pass to the pcolor function.
                title (str, optional):
                    A title for the plot. Defaults to None.

        Returns:
            plt.Axes:
                A matplotlib Axes object.

        """
        if ax is None:
            ax = plt.gca()

        # This now uses series by default
        # get the header form the series object

        if other_data is None:
            # arrows wil be wind speed data, regardless of other_data input
            other_data = self.wind_speed_series
        other_data_for_arrows = self.wind_speed_series

        df = self.month_hour_mean_matrix(other_data=other_data)
        df_for_arrows = self.month_hour_mean_matrix(other_data=other_data_for_arrows)

        try:
            other_data_header = to_pandas(other_data.name)
        except Exception:
            other_data_header = to_ladybug(other_data.name)

        _wind_directions = df[df.columns.get_level_values(0)[0]]
        _other_data = df[df.columns.get_level_values(0)[-1]]
        _other_data_for_arrows = df_for_arrows[
            df_for_arrows.columns.get_level_values(0)[-1]
        ]

        cmap = kwargs.pop("cmap", "YlGnBu")
        vmin = kwargs.pop("vmin", _other_data.values.min())
        vmax = kwargs.pop("vmax", _other_data.values.max())
        unit = kwargs.pop("unit", other_data_header.unit)
        title = kwargs.pop("title", self.location.source)
        norm = kwargs.pop("norm", plt.Normalize(vmin=vmin, vmax=vmax, clip=True))
        mapper = kwargs.pop("mapper", ScalarMappable(norm=norm, cmap=cmap))
        arrow_scale = kwargs.pop("arrow_scale", 0.8)
        pc = ax.pcolor(_other_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        
        if show_arrows:
            _x, _y = -np.array(
                azimuth_to_vector(_wind_directions.values)
            )  # negated to get the direction the wind is blowing TO
            
            ax.quiver(
                np.arange(1, 13, 1) - 0.5,
                np.arange(0, 24, 1) + 0.5,
                (_x * _other_data_for_arrows.values / 2) * arrow_scale,
                (_y * _other_data_for_arrows.values / 2) * arrow_scale,
                pivot="mid",
                fc="white",
                ec="black",
                lw=0.5,
                alpha=0.5,
            )

        if show_values:
            for _xx, col in enumerate(_wind_directions.values.T):
                for _yy, _ in enumerate(col.T):
                    local_value = _other_data.values[_yy, _xx]
                    cell_color = mapper.to_rgba(local_value)
                    text_color = contrasting_color(cell_color)
                    # direction text
                    ax.text(
                        _xx,
                        _yy,
                        f"{_wind_directions.values[_yy][_xx]:0.0f}$\degree$",
                        color=text_color,
                        ha="left",
                        va="bottom",
                        fontsize="xx-small",
                    )
                    # other_data text
                    ax.text(
                        _xx + 1,
                        _yy + 1,
                        f"{_other_data.values[_yy][_xx]:0.1f}{unit}",
                        color=text_color,
                        ha="right",
                        va="top",
                        fontsize="xx-small",
                    )

        # add title and colorbar
        ax.set_title(title)
        ax.set_xticks([i - 0.5 for i in range(1, 13, 1)])
        ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13, 1)])
        ax.set_yticks([i + 0.5 for i in range(24)])
        ax.set_yticklabels([f"{i:02d}:00" for i in range(24)])
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)

        cb = plt.colorbar(pc, label=unit, pad=0.01)
        cb.outline.set_visible(False)

        return ax

    def plot_windrose(
        self,
        ax: Optional[Axes] = None,
        directions: int = 36,
        other_data: Optional[list[float]] = None,
        other_bins: Union[list[Union[float, int]], int] = 11,
        show_legend: bool = True,
        show_label: bool = False,
        remove_calm: bool = True,
        **kwargs,
    ) -> Axes:
        """Create a wind rose showing wind speed and direction frequency.

        Args:
            ax (plt.Axes, optional):
                The axes to plot this chart on. Defaults to None.
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction.
                If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.
                If other data is None, then the default Beaufort bins will be used,
                otherwise 11 evenly spaced bins will be used.
            show_legend (bool, optional):
                Whether to show the legend.
                Defaults to True.
            show_label (bool, optional):
                Whether to show the bin labels.
                Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to the plot.

        Returns:
            plt.Axes: The axes object.

        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        calm_wind_speeds = np.array(self.calm_mask(threshold=0.01))

        if other_data is None:
            other_data = self.wind_speed

        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as wind data")

        # obtain kwarg data
        cmap = kwargs.pop("cmap", "YlGnBu")
        title = kwargs.pop(
            "title",
            f"{self.location.source}"
            + (
                f" ({sum(calm_wind_speeds) / len(self):0.2%} calm)"
                if remove_calm
                else ""
            )
        )

        # create grouped data for plotting
        binned = self.filter_by_boolean_mask((~calm_wind_speeds).tolist()).histogram(
            directions=directions,
            other_data=np.array(other_data)[~calm_wind_speeds].tolist(),
            other_bins=other_bins,
            density=True,
        )

        ylim = kwargs.pop("ylim", (0, max(binned.sum(axis=1))))
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple of length 2.")

        # obtain colors
        if not isinstance(cmap, Colormap):
            cmap = plt.get_cmap(cmap)
        colors = [to_colour(cmap(i), fmt="hex") for i in np.linspace(0, 1, len(binned.columns))]

        # create the patches
        theta_width = np.deg2rad(360 / directions)
        patches = []
        color_list = []
        x = theta_width / 2
        for _, data_values in binned.iterrows():
            y = 0
            for n, val in enumerate(data_values.values):
                patches.append(
                    Rectangle(
                        xy=(x, y),
                        width=theta_width,
                        height=val,
                        alpha=1,
                    )
                )
                color_list.append(colors[n])
                y += val
            if show_label:
                ax.text(x, y, f"{y:0.1%}", ha="center", va="center", fontsize="x-small")
            x += theta_width
        local_cmap = ListedColormap(np.array(color_list).flatten())
        pc = PatchCollection(patches, cmap=local_cmap)
        pc.set_array(np.arange(len(color_list)))
        ax.add_collection(pc)

        # construct legend
        if show_legend:
            handles = [
                Patch(color=colors[n], label=col)
                for n, col in enumerate(binned.columns)
            ]
            _ = ax.legend(
                handles=handles,
                bbox_to_anchor=(1.1, 0.5),
                loc="center left",
                ncol=1,
                borderaxespad=0,
                frameon=False,
                fontsize="small",
                title=binned.columns.name,
                title_fontsize="small",
            )

        # set y-axis limits
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        # format the plot
        ax.set_title(title)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.spines["polar"].set_visible(False)
        ax.grid(True, which="both", ls="--", zorder=0, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.get_yticklabels(), fontsize="small")
        ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
        ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
        ax.set_xticks(
            np.radians(
                (
                    22.5,
                    45,
                    67.5,
                    112.5,
                    135,
                    157.5,
                    202.5,
                    225,
                    247.5,
                    292.5,
                    315,
                    337.5,
                )
            ),
            minor=True,
        )
        ax.set_xticklabels(
            (
                "NNE",
                "NE",
                "ENE",
                "ESE",
                "SE",
                "SSE",
                "SSW",
                "SW",
                "WSW",
                "WNW",
                "NW",
                "NNW",
            ),
            minor=True,
            **{"fontsize": "x-small"},
        )

        return ax

    def plot_default_windroses(
        self,
        output_directory: Path,
        figsize: tuple[float, float] = (10, 10),
        directions=16,
    ) -> dict[str, Any]:
        # get min/max wind speed
        _min = 0
        _max = self.wind_speed_series.max()
        bins = np.linspace(_min, _max, 11).tolist()

        d = {}

        def plot_and_save(ap):
            temp_self = self.filter_by_analysis_period(ap.value)
            fig, ax = plt.subplots(
                1, 1, figsize=figsize, subplot_kw={"projection": "polar"}
            )
            temp_self.plot_windrose(
                ax=ax,
                directions=directions,
                other_data=temp_self.wind_speed,
                other_bins=bins,
                show_legend=True,
                show_label=False,
                remove_calm=True,
            )
            lgd = ax.get_legend()
            lgd.set_title("m/s")
            ax.set_title(
                f"{analysis_period_to_string(ap.value)} ({temp_self.proportion_calm():0.1%} calm)"
            )
            sp = output_directory / f"windrose_{ap.name.lower()}.png"
            fig.savefig(
                sp,
                bbox_inches="tight",
            )
            plt.close(fig)
            return ap.name, sp

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(plot_and_save, DefaultAnalysisPeriod))

        d = dict(results)
        return d

    def plot_windhistogram(
        self,
        ax: Optional[Axes] = None,
        directions: int = 36,
        other_data: Union[list[float], None] = None,
        other_bins: Union[list[float], int] = 11,
        density: bool = False,
        cmap: Union[str, Colormap] = "YlGnBu",
        show_values: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Axes:
        """Plot a 2D-histogram for a collection of wind speeds and directions.

        Args:
            ax (plt.Axes, optional):
                The axis to plot results on. Defaults to None.
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.
            density (bool, optional):
                If True, then return the probability density function. Defaults to False.
            cmap (Union[str, Colormap], optional):
                The colormap to use. Defaults to "YlGnBu".
            show_values (bool, optional):
                Whether to show values in the cells. Defaults to True.
            vmin (float, optional):
                The minimum value for the colormap. Defaults to None.
            vmax (float, optional):
                The maximum value for the colormap. Defaults to None.

        Returns:
            plt.Axes:
                A matplotlib Axes object.

        """
        # FIXME - This method kind-of works, but needs to be fixed to properly work!

        if ax is None:
            ax = plt.gca()

        hist = self.histogram(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
            density=density,
        )

        vmin = hist.values.min() if vmin is None else vmin
        vmax = hist.values.max() if vmax is None else vmax
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = ScalarMappable(norm=norm, cmap=cmap)

        _xticks = np.roll(hist.index, 1)
        _values = np.roll(hist.values, 1, axis=0).T

        pc = ax.pcolor(_values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(0.5, len(hist.index), 1), labels=_xticks, rotation=90)
        ax.set_xlabel(hist.index.name)
        ax.set_yticks(np.arange(0.5, len(hist.columns), 1), labels=hist.columns)
        ax.set_ylabel(hist.columns.name)

        cb = plt.colorbar(pc, pad=0.01, label="Density" if density else "Count")
        if density:
            cb.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        cb.outline.set_visible(False)

        ax.set_title(self.location.source)

        if show_values:
            for _xx, row in enumerate(_values):
                for _yy, col in enumerate(row):
                    if (col * 100).round(1) == 0:
                        continue
                    cell_color = mapper.to_rgba(col)
                    text_color = contrasting_color(cell_color)
                    ax.text(
                        _yy + 0.5,
                        _xx + 0.5,
                        f"{col:0.2%}" if density else col,
                        color=text_color,
                        ha="center",
                        va="center",
                        fontsize="xx-small",
                    )

        return ax

    def plot_densityfunction(
        self,
        ax: Optional[Axes] = None,
        speed_bins: Union[list[float], int] = 11,
        percentiles: tuple[float, ...] = (0.5, 0.95),
        function: str = "pdf",
        ylim: Union[tuple[float, float], None] = None,
        **kwargs,
    ) -> Axes:
        """Create a histogram showing wind speed frequency.

        Args:
            ax (plt.Axes, optional):
                The axes to plot this chart on. Defaults to None.
            speed_bins (list[float], optional):
                The wind speed bins to use for the histogram. These bins are right inclusive.
            percentiles (tuple[float], optional):
                The percentiles to plot. Defaults to (0.5, 0.95).
            function (str, optional):
                The function to use. Either "pdf" or "cdf". Defaults to "pdf".
            ylim (tuple[float], optional):
                The y-axis limits. Defaults to None.

        Returns:
            plt.Axes: The axes object.

        """
        # FIXME - this method kind of works, but could be done better

        if function not in ["pdf", "cdf"]:
            raise ValueError('function must be either "pdf" or "cdf".')

        if ax is None:
            ax = plt.gca()

        ax.set_title(
            f"{str(self)}\n{'Probability Density Function' if function == 'pdf' else 'Cumulative Density Function'}"
        )

        self.wind_speed_series.plot.hist(
            ax=ax,
            density=True,
            bins=speed_bins,
            cumulative=True if function == "cdf" else False,
            **kwargs,
        )

        for percentile in percentiles:
            x = np.quantile(self.wind_speed_series, percentile)
            ax.axvline(x, 0, 1, ls="--", lw=1, c="black", alpha=0.5)
            ax.text(
                x + 0.05,
                0,
                f"{percentile:0.0%}\n{x:0.2f}m/s",
                ha="left",
                va="bottom",
            )

        ax.set_xlim(0, ax.get_xlim()[-1])
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Frequency")

        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))

        return ax

    # endregion: visualization

    # region: UsefulThings

    def wind_exposure(
        self,
        shades: tuple[Shade, ...],
        origin: Point3D = Point3D(0, 0, 1.2),  # type: ignore
        edge_acceleration_width: float = 0.0,
        edge_acceleration_factor: float = 1.2,
        parallel: bool = False,
    ) -> tuple[float, ...]:
        """Calculate annual hourly wind exposure.

        - 0 means the wind is blocked
        - 1 means it is visible
        - 0.25 mean the wind is only 25% 'visible' (through a 75% porous medium)
        - >1 means the wind is accelerated (e.g., around an edge)

        Args:
            shades (tuple[Shade, ...]):
                A set of shades that may include temporal transmissivity
                properties.
            origin (Point3D, optional):
                The location of the sensor from which sun exposure is
                calculated, in relation to the shading objects.
            edge_acceleration_width (float, optional):
                The distance from the edge of a shade object where wind
                acceleration effects begin. Defaults to 0.0 (i.e., no edge
                acceleration effects).
            edge_acceleration_factor (float, optional):
                The factor by which wind speed is increased near the edge of
                a shade object. Defaults to 1.2.
            parallel (bool, optional):
                If True, run in parallel. Useful when number of shade objects
                is high, otherwise the overheads of set-up aren't worth it.
        Returns:
            List[float]:
                A list of annual hourly values denoting sun exposure values.

        Examples
        --------
        >>> from lbttk.plot.matplotlib import heatmap
        >>> from lbttk.wind.wind import Wind
        >>> from ladybug_geometry.geometry3d import Point3D, Face3D, LineSegment3D, Vector3D, Plane
        >>> from honeybee.shade import Shade
        >>> import numpy as np
        >>> import pandas as pd
        >>> from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval

        >>> w = Wind.from_epw("./file.epw")

        >>> shd_dynamic_south = Shade(
        >>>     "shade",
        >>>     geometry=Face3D.from_extrusion(
        >>>         LineSegment3D.from_end_points(Point3D(-8, -5, 0), Point3D(8, -5, 0)),
        >>>         Vector3D(0, 0, 25),
        >>>     ),
        >>> )
        >>> shd_dynamic_south.properties.energy.transmittance_schedule = ScheduleFixedInterval(
        >>>     identifier="transmissivity", values=abs((np.arange(8760) - 4380) / 4380).tolist()
        >>> )

        >>> shd_opaque_north = Shade(
        >>>     identifier="shade_north",
        >>>     Face3D.from_extrusion(
        >>>         LineSegment3D.from_end_points(Point3D(-8, 5, 0), Point3D(8, 5, 0)),
        >>>         Vector3D(0, 0, 25),
        >>>     ),
        >>> )

        >>> result = w.wind_exposure(
        >>>     shades=(shd_dynamic_south, shd_opaque_north),
        >>>     origin=Point3D(0, 0, 1.2),
        >>>     parallel=False,
        >>> )

        >>> heatmap(pd.Series(result, name="hi", index=w.datetimes))
        """

        # TODO - add ege acceleration effects

        # validate inputs
        if not all(isinstance(i, Shade) for i in shades):
            raise TypeError("shades must be an iterable of ladybug Shade objects.")

        # create shade transmissivities
        shade_transmissivities = []
        for hb_shd in shades:
            try:
                shade_transmissivities.append(
                    get_schedule_as_data_collection(
                        hb_shd.properties.energy.transmittance_schedule  # type: ignore
                    ).values
                )
            except ValueError:
                # no schedule available, assume fully opaque
                shade_transmissivities.append(tuple([0.0] * len(self)))

        if not parallel:
            # for each sun, determine whether a ray from the origin to the sun intersects
            # any of the shade objects, and the resultant visibility as product of the
            # intersecting shade transmissivities at that time

            wind_visibility = []
            for n, direction in enumerate(self.wind_direction):
                if edge_acceleration_width == 0:
                    ray = Ray3D(origin, Vector3D(*azimuth_to_vector(direction + 180)))
                    transmissivity = 1.0
                    for m, hb_shd in enumerate(shades):
                        # check for intersection
                        if hb_shd.geometry.intersect_line_ray(ray):
                            # wind is blocked by shade geometry
                            transmissivity = min(shade_transmissivities[m][n], transmissivity)
                else:
                    ray1 = Ray3D(origin, Vector3D(*azimuth_to_vector(direction + 180 - (edge_acceleration_width / 2))))
                    ray2 = Ray3D(origin, Vector3D(*azimuth_to_vector(direction + 180 + (edge_acceleration_width / 2))))
                    transmissivity = 1.0
                    for m, hb_shd in enumerate(shades):
                        # check for intersection
                        intersect1 = hb_shd.geometry.intersect_line_ray(ray1)
                        intersect2 = hb_shd.geometry.intersect_line_ray(ray2)
                        if intersect1 or intersect2:
                            # wind is blocked by shade geometry
                            local_transmissivity = shade_transmissivities[m][n]
                            if intersect1 and intersect2:
                                transmissivity = min(local_transmissivity, transmissivity)
                            else:
                                transmissivity *= edge_acceleration_factor
                wind_visibility.append(transmissivity)
        else:

            def process_direction(n: int, direction: float, edge_acceleration_width: float, edge_acceleration_factor: float) -> tuple[int, float]:
                """Process a single wind direction and return its 'visibility'.
                """
                if edge_acceleration_factor == 0:
                    ray = Ray3D(origin, Vector3D(*azimuth_to_vector(direction + 180)))
                    transmissivity = 1.0

                    for m, hb_shd in enumerate(shades):
                        # check for intersection
                        if hb_shd.geometry.intersect_line_ray(ray):
                            # sun is blocked by shade geometry
                            transmissivity = min(
                                shade_transmissivities[m][n], transmissivity
                            )

                    return (n, transmissivity)
                else:
                    ray1 = Ray3D(origin, Vector3D(*azimuth_to_vector(direction + 180 - (edge_acceleration_width / 2))))
                    ray2 = Ray3D(origin, Vector3D(*azimuth_to_vector(direction + 180 + (edge_acceleration_width / 2))))
                    transmissivity = 1.0

                    for m, hb_shd in enumerate(shades):
                        # check for intersection
                        intersect1 = hb_shd.geometry.intersect_line_ray(ray1)
                        intersect2 = hb_shd.geometry.intersect_line_ray(ray2)
                        if intersect1 or intersect2:
                            # sun is blocked by shade geometry
                            local_transmissivity = shade_transmissivities[m][n]
                            if intersect1 and intersect2:
                                transmissivity = min(local_transmissivity, transmissivity)
                            else:
                                transmissivity *= edge_acceleration_factor

                    return (n, transmissivity)

            # Process all directions in parallel
            wind_visibility: list[float] = [None] * len(self)  # type: ignore

            with ThreadPoolExecutor(max_workers=min(len(self), 8)) as executor:
                # Submit all sun calculations
                futures = {
                    executor.submit(process_direction, n, direction): n
                    for n, direction in enumerate(self.wind_direction)
                }

                # Collect results in order
                for future in as_completed(futures):
                    idx, visibility = future.result()
                    wind_visibility[idx] = visibility

        return tuple(wind_visibility)
    
    # endregion: UsefulThings