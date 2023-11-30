"""Methods for working with time-indexed wind data."""

# pylint: disable=E0401
import calendar
from cgitb import lookup
import json

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# pylint: enable=E0401

import matplotlib.patches as mpatches
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.dt import DateTime
from ladybug.windrose import WindRose
from ladybug.epw import EPW
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from tqdm import tqdm

from .bhom.analytics import bhom_analytics
from .categorical.categories import BEAUFORT_CATEGORIES
from .directionbins import DirectionBins
from .helpers import (
    OpenMeteoVariable,
    angle_from_north,
    angle_to_vector,
    cardinality,
    circular_weighted_mean,
    rolling_window,
    scrape_meteostat,
    scrape_openmeteo,
    weibull_directional,
    weibull_pdf,
    wind_speed_at_height,
    chunks,
)
from .ladybug_extension.analysisperiod import (
    AnalysisPeriod,
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
)
from .plot._timeseries import timeseries
from .plot.utilities import contrasting_color, format_polar_plot

from ladybug.datacollection import HourlyContinuousCollection, BaseCollection


@dataclass(init=True, eq=True, repr=True)
class WindData:
    """An object containing historic, time-indexed wind data.

    Args:
        wind_speeds (list[int | float | np.number]):
            An iterable of wind speeds in m/s.
        wind_directions (list[int | float | np.number]):
            An iterable of wind directions in degrees from North (with North at 0-degrees).
        datetimes (Union[pd.DatetimeIndex, list[Union[datetime, np.datetime64, pd.Timestamp]]]):
            An iterable of datetime-like objects.
        height_above_ground (float, optional):
            The height above ground (in m) where the input wind speeds and directions were collected. Defaults to 10m.
        source (str, optional):
                A source string to describe where the input data comes from. Defaults to None.
    """

    wind_speeds: list[float]
    wind_directions: list[float]
    datetimes: list[datetime] | pd.DatetimeIndex
    height_above_ground: float
    source: str = None

    def __post_init__(self):
        if self.height_above_ground < 0.1:
            raise ValueError("Height above ground must be >= 0.1m.")

        if not (
            len(self.wind_speeds) == len(self.wind_directions) == len(self.datetimes)
        ):
            raise ValueError(
                "wind_speeds, wind_directions and datetimes must be the same length."
            )

        if len(self.wind_speeds) <= 1:
            raise ValueError(
                "wind_speeds, wind_directions and datetimes must be at least 2 items long."
            )

        if len(set(self.datetimes)) != len(self.datetimes):
            raise ValueError("datetimes contains duplicates.")

        # convert to lists
        self.wind_speeds = np.array(self.wind_speeds)
        self.wind_directions = np.array(self.wind_directions)
        self.datetimes = pd.DatetimeIndex(self.datetimes)

        # validate wind speeds and directions
        if np.any(np.isnan(self.wind_speeds)):
            raise ValueError("wind_speeds contains null values.")

        if np.any(np.isnan(self.wind_directions)):
            raise ValueError("wind_directions contains null values.")

        if np.any(self.wind_speeds < 0):
            raise ValueError("wind_speeds must be >= 0")
        if np.any(self.wind_directions < 0) or np.any(self.wind_directions > 360):
            raise ValueError("wind_directions must be within 0-360")
        self.wind_directions = self.wind_directions % 360

    def __len__(self) -> int:
        return len(self.datetimes)

    def __repr__(self) -> str:
        """The printable representation of the given object"""
        if self.source:
            return f"{self.__class__.__name__}(@{self.height_above_ground}m) from {self.source}"

        return (
            f"{self.__class__.__name__}({min(self.datetimes):%Y-%m-%d} to "
            f"{max(self.datetimes):%Y-%m-%d}, n={len(self.datetimes)} @{self.freq}, "
            f"@{self.height_above_ground}m) NO SOURCE"
        )

    def __str__(self) -> str:
        """The string representation of the given object"""
        return self.__repr__()

    #################
    # CLASS METHODS #
    #################

    def to_dict(self) -> dict:
        """Return the object as a dictionary."""

        return {
            "_t": "BH.oM.LadybugTools.Wind",
            "wind_speeds": [float(i) for i in self.wind_speeds],
            "wind_directions": [float(i) for i in self.wind_directions],
            "datetimes": [i.isoformat() for i in self.datetimes],
            "height_above_ground": self.height_above_ground,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WindData":
        """Create a DirectionBins object from a dictionary."""

        return cls(
            wind_speeds=d["wind_speeds"],
            wind_directions=d["wind_directions"],
            datetimes=pd.to_datetime(d["datetimes"]),
            height_above_ground=d["height_above_ground"],
            source=d["source"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "WindData":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Convert this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "WindData":
        """Create this object from a JSON file."""
        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        wind_speed_column: Any,
        wind_direction_column: Any,
        height_above_ground: float = 10,
        source: str = "DataFrame",
    ) -> "WindData":
        """Create a Wind object from a Pandas DataFrame, with WindSpeed and WindDirection columns.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing speed and direction columns, and a datetime index.
            wind_speed_column (str):
                The name of the column where wind-speed data exists.
            wind_direction_column (str):
                The name of the column where wind-direction data exists.
            height_above_ground (float, optional):
                Defaults to 10m.
            source (str, optional):
                A source string to describe where the input data comes from. Defaults to "DataFrame"".
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df must be of type {type(pd.DataFrame)}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"The DataFrame's index must be of type {type(pd.DatetimeIndex)}"
            )

        # remove NaN values
        df.dropna(axis=0, how="any", inplace=True)

        # remove duplicates in input dataframe
        df = df.loc[~df.index.duplicated()]

        return cls(
            wind_speeds=df[wind_speed_column].tolist(),
            wind_directions=df[wind_direction_column].tolist(),
            datetimes=df.index.tolist(),
            height_above_ground=height_above_ground,
            source=source,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        wind_speed_column: str,
        wind_direction_column: str,
        height_above_ground: float = 10,
        **kwargs,
    ) -> "WindData":
        """Create a Wind object from a csv containing wind speed and direction columns.

        Args:
            csv_path (Path):
                The path to the CSV file containing speed and direction columns, and a datetime index.
            wind_speed_column (str):
                The name of the column where wind-speed data exists.
            wind_direction_column (str):
                The name of the column where wind-direction data exists.
            height_above_ground (float, optional):
                Defaults to 10m.
            **kwargs:
                Additional keyword arguments passed to pd.read_csv.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, **kwargs)
        return cls.from_dataframe(
            df,
            wind_speed_column=wind_speed_column,
            wind_direction_column=wind_direction_column,
            height_above_ground=height_above_ground,
            source=csv_path.name,
        )

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "WindData":
        """Create a Wind object from an EPW file or object.

        Args:
            epw (Path | EPW):
                The path to the EPW file, or an EPW object.
        """

        if isinstance(epw, (str, Path)):
            source = Path(epw).name
            epw = EPW(epw)
        else:
            source = Path(epw.file_path).name

        return cls(
            wind_speeds=epw.wind_speed.values,
            wind_directions=epw.wind_direction.values,
            datetimes=analysis_period_to_datetimes(AnalysisPeriod()),
            height_above_ground=10,
            source=source,
        )

    @classmethod
    def from_openmeteo(
        cls,
        latitude: float,
        longitude: float,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> "WindData":
        """Create a Wind object from data obtained from the Open-Meteo database of historic weather station data.

        Args:
            latitude (float):
                The latitude of the target site, in degrees.
            longitude (float):
                The longitude of the target site, in degrees.
            start_date (datetime | str):
                The start-date from which records will be obtained.
            end_date (datetime | str):
                The end-date beyond which records will be ignored.
        """

        df = scrape_openmeteo(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=[
                OpenMeteoVariable.WINDSPEED_10M,
                OpenMeteoVariable.WINDDIRECTION_10M,
            ],
            convert_units=True,
        )

        df.dropna(how="any", axis=0, inplace=True)
        wind_speeds = df["Wind Speed (m/s)"].tolist()
        wind_directions = df["Wind Direction (degrees)"].tolist()
        if len(wind_speeds) == 0 or len(wind_directions) == 0:
            raise ValueError(
                "OpenMeteo did not return any data for the given latitude, longitude and start/end dates."
            )
        datetimes = df.index.tolist()

        return cls(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            datetimes=datetimes,
            height_above_ground=10,
            source="OpenMeteo",
        )

    @classmethod
    def from_meteostat(
        cls,
        latitude: float,
        longitude: float,
        start_date: datetime | str,
        end_date: datetime | str,
        altitude: float = 10,
    ) -> "WindData":
        """Create a Wind object from data obtained from the Meteostat database of historic weather station data.

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
                The altitude of the target site, in meters. Defaults to 10.
        """

        df = scrape_meteostat(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            altitude=altitude,
            convert_units=True,
        )[["Wind Speed (m/s)", "Wind Direction (degrees)"]]

        df.dropna(axis=0, inplace=True, how="any")

        return cls.from_dataframe(
            df=df,
            wind_speed_column="Wind Speed (m/s)",
            wind_direction_column="Wind Direction (degrees)",
            height_above_ground=altitude,
            source="Meteostat",
        )

    @classmethod
    def from_average(
        cls, wind_objects: list["WindData"], weights: list[float] = None
    ) -> "WindData":
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""

        # create default weightings if None
        if weights is None:
            weights = [1 / len(wind_objects)] * len(wind_objects)
        else:
            if sum(weights) != 1:
                raise ValueError("weights must total 1.")

        # create source string
        source = []
        for src, wgt in list(zip([wind_objects, weights])):
            source.append(f"{src.source}|{wgt}")
        source = "_".join(source)

        # align collections so that intersection only is created
        df_ws = pd.concat([i.ws for i in wind_objects], axis=1).dropna()
        df_wd = pd.concat([i.wd for i in wind_objects], axis=1).dropna()

        # construct the weighted means
        wd_avg = np.array(
            [circular_weighted_mean(i, weights) for _, i in df_wd.iterrows()]
        )
        ws_avg = np.average(df_ws, axis=1, weights=weights)
        dts = df_ws.index

        # return the new averaged object
        return cls(
            wind_speeds=ws_avg.tolist(),
            wind_directions=wd_avg.tolist(),
            datetimes=dts,
            height_above_ground=np.average(
                [i.height_above_ground for i in wind_objects], weights=weights
            ),
            source=source,
        )

    @classmethod
    def from_uv(
        cls,
        u: list[float],
        v: list[float],
        datetimes: list[datetime],
        height_above_ground: float = 10,
        source: str = None,
    ) -> "WindData":
        """Create a Wind object from a set of U, V wind components.

        Args:
            u (list[float]):
                An iterable of U (eastward) wind components in m/s.
            v (list[float]):
                An iterable of V (northward) wind components in m/s.
            datetimes (list[datetime]):
                An iterable of datetime-like objects.
            height_above_ground (float, optional):
                The height above ground (in m) where the input wind speeds and directions were collected.
                Defaults to 10m.
            source (str, optional):
                A source string to describe where the input data comes from. Defaults to None.

        Returns:
            Wind:
                A Wind object!
        """

        # convert UV into angle and magnitude
        wind_direction = angle_from_north(np.stack([u, v]))
        wind_speed = np.sqrt(np.square(u) + np.square(v))

        if any(wind_direction[wind_speed == 0] == 90):
            warnings.warn(
                "Some input vectors have velocity of 0. This is not bad, but can mean directions may be misreported."
            )

        return cls(
            wind_speeds=wind_speed.tolist(),
            wind_directions=wind_direction.tolist(),
            datetimes=datetimes,
            height_above_ground=height_above_ground,
            source=source,
        )

    # ##############
    # # PROPERTIES #
    # ##############

    @property
    def freq(self) -> str:
        """Return the inferred frequency of the datetimes associated with this object."""
        freq = pd.infer_freq(self.datetimes)
        if freq is None:
            return "inconsistent"
        return freq

    @property
    def index(self) -> pd.DatetimeIndex:
        """Get the datetimes as a pandas DateTimeIndex."""
        return pd.to_datetime(self.datetimes)

    @property
    def ws(self) -> pd.Series:
        """Convenience accessor for wind speeds as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_speeds, index=self.index, name="Wind Speed (m/s)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def wd(self) -> pd.Series:
        """Convenience accessor for wind directions as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_directions, index=self.index, name="Wind Direction (degrees)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def df(self) -> pd.DataFrame:
        """Convenience accessor for wind speed and direction as a time-indexed pd.DataFrame object."""
        return pd.concat([self.ws, self.wd], axis=1)

    @property
    def calm_datetimes(self) -> list[datetime]:
        """Return the datetimes where wind speed is < 0.1.

        Returns:
            list[datetime]:
                "Calm" wind datetimes.
        """
        return self.ws[self.ws <= 0.1].index.tolist()  # pylint: disable=E1136

    @property
    def uv(self) -> pd.DataFrame:
        """Return the U and V wind components in m/s."""
        u, v = angle_to_vector(self.wd)
        return pd.concat([u * self.ws, v * self.ws], axis=1, keys=["u", "v"])

    @property
    def mean_uv(self) -> list[float, float]:
        """Calculate the average U and V wind components in m/s.

        Returns:
            list[float, float]:
                A tuple containing the average U and V wind components.
        """
        return self.uv.mean().tolist()

    @property
    def mean_speed(self) -> float:
        """Return the mean wind speed for this object."""
        return np.linalg.norm(self.mean_uv)

    @property
    def mean_direction(self) -> tuple[float, float]:
        """Calculate the average speed and direction for this object.

        Returns:
            tuple[float, float]:
                A tuple containing the average speed and direction.
        """
        return angle_from_north(self.mean_uv)

    @property
    def min_speed(self) -> float:
        """Return the min wind speed for this object."""
        return self.ws.min()

    @property
    def max_speed(self) -> float:
        """Return the max wind speed for this object."""
        return self.ws.max()

    @property
    def median_speed(self) -> float:
        """Return the median wind speed for this object."""
        return self.ws.median()

    ###################
    # GENERAL METHODS #
    ###################

    def calm(self, threshold: float = 1e-10) -> float:
        """Return the proportion of timesteps "calm" (i.e. wind-speed ≤ 1e-10)."""
        return (self.ws < threshold).sum() / len(self.ws)

    def percentile(self, percentile: float) -> float:
        """Calculate the wind speed at the given percentile.

        Args:
            percentile (float):
                The percentile to calculate.

        Returns:
            float:
                Wind speed at the given percentile.
        """
        return self.ws.quantile(percentile)

    def resample(self, rule: pd.DateOffset | pd.Timedelta | str) -> "WindData":
        """Resample the wind data collection to a different timestep. This can only be used to downsample.

        Args:
            rule (Union[pd.DateOffset, pd.Timedelta, str]):
                A rule for resampling. This uses the same inputs as a Pandas Series.resample() method.

        Returns:
            WindData:
                A wind data collection object!
        """

        warnings.warn(
            (
                "Resampling wind speeds and direction is generally not advisable. "
                "When input directions are opposing, the average returned is likely inaccurate, "
                "and the average speed does not include any outliers. USE WITH CAUTION!"
            )
        )

        common_dt = pd.to_datetime("2000-01-01")
        f_a = common_dt + to_offset(self.freq)
        f_b = common_dt + to_offset(rule)
        if f_a < f_b:
            raise ValueError("Resampling can only be used to downsample.")

        resampled_speeds = self.ws.resample(rule).mean()
        resampled_datetimes = resampled_speeds.index.tolist()
        resampled_directions = self.wd.resample(rule).apply(circular_weighted_mean)

        return WindData(
            wind_speeds=resampled_speeds.tolist(),
            wind_directions=resampled_directions.tolist(),
            datetimes=resampled_datetimes,
            height_above_ground=self.height_above_ground,
            source=self.source,
        )

    def to_height(
        self,
        target_height: float,
        terrain_roughness_length: float = 1,
        log_function: bool = True,
    ) -> "WindData":
        """Translate the object to a different height above ground.

        Args:
            target_height (float):
                Height to translate to (in m).
            terrain_roughness_length (float, optional):
                Terrain roughness (how big objects are to adjust translation). Defaults to 1.
            log_function (bool, optional):
                Whether to use log-function or pow-function. Defaults to True.

        Returns:
            Wind:
                A translated Wind object.
        """
        ws = wind_speed_at_height(
            reference_value=self.ws,
            reference_height=self.height_above_ground,
            target_height=target_height,
            terrain_roughness_length=terrain_roughness_length,
            log_function=log_function,
        )
        return WindData(
            wind_speeds=ws.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=target_height,
            source=f"{self.source} translated to {target_height}m",
        )

    def apply_directional_factors(
        self, directions: int, factors: tuple[float]
    ) -> "WindData":
        """Adjust wind speed values by a set of factors per direction.

        Args:
            directions (int):
                The number of directions to bin wind-directions into.
            factors (tuple[float], optional):
                Adjustment factors per direction. Defaults to (i for i in range(8)).

        Returns:
            Wind:
                An adjusted Wind object.
        """

        factors = np.array(factors).tolist()

        if len(factors) != directions:
            raise ValueError(
                f"number of factors must equal number of directions ({directions} != {len(factors)})"
            )

        bin_edges = _direction_bin_edges(directions=directions)
        cutted = pd.cut(self.wd, np.unique(bin_edges), right=True)
        raise NotImplementedError("dont work here yet")
        # TODO - from here re-add functions
        lookup = dict(zip(a.cat.categories.tolist(), factors))

        factors = []
        for idx, (ws, wd) in zip([self.ws, self.wd]):
            if wd > bin_edges[0][0] | wd <= bin_edges[0][1]:
                factors.append(factors[0])
            else:
                factors.append(lookup[bin_edges[1:][wd]])

        # repeat first element in inputs to apply to the two directions centered about North
        factors = factors + [factors[0]]  # type: ignore

        # create direction bins for factor to be applied
        intervals = direction_bins.interval_index

        # bin data and get factor indices to apply to each value
        binned = pd.cut(self.wd, bins=intervals, include_lowest=True, right=False)
        d = {i: n for n, i in enumerate(intervals)}
        factor_indices = (
            binned.map(d).fillna(0).values.tolist()
        )  # filling NaN with 0 as this is the value at which failure happen (0/360 doesn't fit into any of the bins)
        all_factors = [factors[int(i)] for i in factor_indices]
        ws = self.ws * all_factors

        return WindData(
            wind_speeds=ws.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod | tuple[AnalysisPeriod],
    ) -> "WindData":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if isinstance(analysis_period, AnalysisPeriod):
            analysis_period = (analysis_period,)

        for ap in analysis_period:
            if ap.timestep != 1:
                raise ValueError("The timestep of the analysis period must be 1 hour.")

        # remove 29th Feb dates where present
        df = self.df
        df = df[~((df.index.month == 2) & (df.index.day == 29))]

        # filter available data
        possible_datetimes = [
            DateTime(dt.month, dt.day, dt.hour, dt.minute) for dt in df.index
        ]
        lookup = dict(
            zip(AnalysisPeriod().datetimes, analysis_period_to_boolean(analysis_period))
        )
        mask = [lookup[i] for i in possible_datetimes]
        df = df[mask]

        if len(df) == 0:
            raise ValueError("No data remains within the given analysis_period filter.")

        return WindData.from_dataframe(
            df,
            wind_speed_column="Wind Speed (m/s)",
            wind_direction_column="Wind Direction (degrees)",
            height_above_ground=self.height_above_ground,
            source=self.source,
        )

    def filter_by_boolean_mask(self, mask: tuple[bool]) -> "WindData":
        """Filter the current object by a boolean mask.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if len(mask) != len(self.ws):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given boolean filters.")

        return WindData(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=self.source,
        )

    def filter_by_time(
        self,
        months: tuple[float] = tuple(range(1, 13, 1)),  # type: ignore
        hours: tuple[int] = tuple(range(0, 24, 1)),  # type: ignore
        years: tuple[int] = tuple(range(1900, 2100, 1)),
    ) -> "WindData":
        """Filter the current object by month and hour.

        Args:
            months (list[int], optional):
                A list of months. Defaults to all months.
            hours (list[int], optional):
                A list of hours. Defaults to all hours.
            years (tuple[int], optional):
                A list of years to include. Default to all years since 1900.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if years is None:
            indices = np.argwhere(
                np.all(
                    [
                        self.datetimes.month.isin(months),
                        self.datetimes.hour.isin(hours),
                    ],
                    axis=0,
                )
            ).flatten()
        else:
            indices = np.argwhere(
                np.all(
                    [
                        self.datetimes.year.isin(years),
                        self.datetimes.month.isin(months),
                        self.datetimes.hour.isin(hours),
                    ],
                    axis=0,
                )
            ).flatten()

        if len(self.ws.iloc[indices]) == 0:
            raise ValueError("No data remains within the given time filters.")

        return WindData(
            wind_speeds=self.ws.iloc[indices].tolist(),
            wind_directions=self.wd.iloc[indices].tolist(),
            datetimes=self.datetimes[indices],
            height_above_ground=self.height_above_ground,
            source=self.source,
        )

    def filter_by_direction(
        self, left_angle: float = 0, right_angle: float = 360, inclusive: bool = True
    ) -> "WindData":
        """Filter the current object by wind direction, based on the angle as observed from a location.

        Args:
            left_angle (float):
                The left-most angle, to the left of which wind speeds and directions will be removed.
            right_angle (float):
                The right-most angle, to the right of which wind speeds and directions will be removed.
            inclusive (bool, optional):
                Include values that are exactly the left or right angle values.

        Return:
            Wind:
                A Wind object!
        """

        if left_angle < 0 or right_angle > 360:
            raise ValueError("Angle limits must be between 0 and 360 degrees.")

        if (left_angle == right_angle) or (left_angle == 360 and right_angle == 0):
            raise ValueError("Angle limits cannot be identical.")

        if left_angle > right_angle:
            if inclusive:
                mask = (self.wd >= left_angle) | (self.wd <= right_angle)
            else:
                mask = (self.wd > left_angle) | (self.wd < right_angle)
        else:
            if inclusive:
                mask = (self.wd >= left_angle) & (self.wd <= right_angle)
            else:
                mask = (self.wd > left_angle) & (self.wd < right_angle)

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given direction filter.")

        return WindData(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=f"{self.source} filtered by direction ({left_angle}-{right_angle})",
        )

    def filter_by_speed(
        self, min_speed: float = 0, max_speed: float = 999, inclusive: bool = True
    ) -> "WindData":
        """Filter the current object by wind speed, based on given low-high limit values.

        Args:
            min_speed (float):
                The lowest speed to include. Values below this wil be removed.
            max_speed (float):
                The highest speed to include. Values above this wil be removed.
            inclusive (bool, optional):
                Include values that are exactly the left or right angle values.

        Return:
            Wind:
                A Wind object!
        """

        if min_speed < 0:
            raise ValueError("min_speed cannot be negative.")

        if max_speed <= min_speed:
            raise ValueError("min_speed must be less than max_speed.")

        if inclusive:
            mask = (self.ws >= min_speed) & (self.ws <= max_speed)
        else:
            mask = (self.ws > min_speed) & (self.ws < max_speed)

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given speed filter.")

        return WindData(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=f"{self.source} filtered by speed ({min_speed}-{max_speed})",
        )

    def prevailing(
        self,
        directions: int = 36,
        n: int = 1,
    ) -> list[float] | list[str]:
        """Calculate the prevailing wind direction/s for this object.

        Args:
            directions (DirectionBins, optional):
                The number of wind directions to bin values into.
            n (int, optional):
                The number of prevailing directions to return. Default is 1.

        Returns:
            list[float] | list[str]:
                A list of wind directions.
        """

        prevailing_angles = (
            self.histogram(directions=directions, midpoint_label=True)
            .sum(axis=0)
            .sort_values(ascending=False)
            .index[:n]
        )
        return prevailing_angles.tolist()

    def probabilities(
        self,
        directions: int = 36,
        percentiles: tuple[float, float] = (0.5, 0.95),
        midpoint_label: bool = False,
    ) -> pd.DataFrame:
        """Calculate the probabilities of wind speeds at the given percentiles, for the direction bins specified.

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.
            percentiles (tuple[float, float], optional):
                A tuple of floats between 0-1 describing percentiles. Defaults to (0.5, 0.95).
            midpoint_label (bool, optional):
                If True, then return the midpoint of each bin as the label. Defaults to False.

        Returns:
            pd.DataFrame:
                A table showing probabilities at the given percentiles.
        """

        bin_edges = _direction_bin_edges(directions)
        direction_labels = (
            _direction_midpoints(directions=directions)
            if midpoint_label
            else [tuple(i) for i in bin_edges]
        )
        binned_data = {}
        for n, (low, high) in enumerate(bin_edges):
            if n == 0:
                binned_data[(low, high)] = self.ws.values[
                    (self.wd > low) | (self.wd <= high)
                ]
            else:
                binned_data[(low, high)] = self.ws.values[
                    (self.wd > low) & (self.wd <= high)
                ]

        probabilities = []
        for percentile in percentiles:
            temp = []
            for _, vals in binned_data.items():
                temp.append(np.quantile(vals, percentile))
            probabilities.append(temp)

        df = pd.DataFrame(
            data=probabilities, columns=direction_labels, index=percentiles  # type: ignore
        )
        return df.T

    def histogram(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
        density: bool = False,
        midpoint_label: bool = False,
    ) -> pd.DataFrame:
        """Bin data by direction, returning counts for each direction.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.
            density (bool, optional):
                If True, then return the probability density function. Defaults to False.
            midpoint_label (bool, optional):
                If True, then return the midpoint of each bin as the label. Defaults to False.

        Returns:
            pd.DataFrame:
                A numpy array, containing the number or probability for each bin, for each direction bin.
        """

        mask = np.ones_like(self.wd.values).astype(bool)
        if other_data is None:
            other_data = self.ws.values
            mask = self.ws > 1e-10
            if other_bins is None:
                other_bins = BEAUFORT_CATEGORIES.bins
        if other_bins is None:
            other_bins = np.linspace(min(other_data), max(other_data), 11)
        other_data = np.array(other_data)

        if len(other_data) != len(self.wd.values):
            raise ValueError(
                "other_data must be the same length as wind direction data."
            )

        return _circular_histogram(
            direction_data=self.wd.values[mask],
            other_data=other_data[mask],
            directions=directions,
            other_bins=other_bins,
            density=density,
            midpoint_label=midpoint_label,
        )

    def cumulative_density_function(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
        density: bool = False,
        midpoint_label: bool = False,
    ) -> pd.DataFrame:
        """Create a table with the cumulative probability density function for each "other_data" and direction.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.
            density (bool, optional):
                If True, then return the probability density function. Defaults to False.
            midpoint_label (bool, optional):
                If True, then return the midpoint of each bin as the label. Defaults to False.

        Returns:
            pd.DataFrame:
                A cumulative density table.
        """
        return self.histogram(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
            density=density,
            midpoint_label=midpoint_label,
        ).cumsum(axis=0)


def _direction_midpoints(directions: int) -> list[float]:
    """Get the list of directions.

    Args:
        directions (int, optional):
            The number of directions to use.

    Returns:
        list[float]:
            The bin bidpoints.
    """

    if directions <= 2:
        raise ValueError("directions must be > 2.")

    return np.linspace(0, 360, directions + 1).tolist()[:-1]


def _direction_bin_edges(directions: int) -> list[list[float]]:
    """Create a list of start/end points for wind directions, each bin increasing from the previous one.
        This method assumes that North is at 0 degrees. The bins returned are all increasing,
        with the one crossing north split into two (and placed at either end of the list of bins).

    Args:
        directions (int, optional):
            The number of directions to bin wind data into.

    Returns:
        list[list[float]]:
            A set of bin edges.
    """

    # get angle for each bin
    bin_angle = 360 / directions

    bin_edges = []
    for d in _direction_midpoints(directions=directions):
        bin_edges.append([(d - (bin_angle / 2)) % 360, (d + (bin_angle / 2)) % 360])

    return bin_edges


def _circular_histogram(
    direction_data: list[float],
    other_data: list[float],
    directions: int,
    other_bins: list[float],
    density: bool = False,
    midpoint_label: bool = True,
) -> pd.DataFrame:
    """Bin data by direction, returning counts for each direction.

    Args:
        direction_data (list[float]):
            An iterable of wind directions in degrees from North (with North at 0-degrees).
        other_data (list[float]):
            An iterable of other data to bin by direction.
        directions (list[list[float]]):
            The direction bins to use for the histogram.
            This should be in the form [[low_0, high_0], [low_1, high_1], ...] and be contiguous, with the first
            element spanning across the north angle at 0. E.g. 324-036. These bins are right inclusive.
        other_bins (list[float]):
            The other data bins to use for the histogram. These bins are right inclusive.
        density (bool, optional):
            If True, then return the probability density function. Defaults to False.
        midpoint_label (bool, optional):
            If True, then return the midpoint of each bin as the label. Defaults to True.

    Returns:
        pd.DataFrame:
            A numpy array, containing the number or probability for each bin, for each direction bin.
    """

    direction_bins = (
        [0] + np.unique(_direction_bin_edges(directions)[1:]).tolist() + [360]
    )
    hist, _, _ = np.histogram2d(
        x=direction_data,
        y=other_data,
        bins=[direction_bins, other_bins],
        density=density,
    )

    direction_labels = (
        _direction_midpoints(directions=directions)
        if midpoint_label
        else [tuple(i) for i in _direction_bin_edges(directions=directions)]
    )
    other_labels = rolling_window(other_bins, 2).T.tolist()

    df = pd.DataFrame(hist.T)
    df[0] = df[0] + df[directions]
    df.drop(columns=[directions], inplace=True)
    df.columns = direction_labels
    df.index = other_labels

    return df
