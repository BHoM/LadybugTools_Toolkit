"""Methods for working with time-indexed wind data."""

# pylint: disable=E0401
import calendar
import json
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.dt import DateTime
from ladybug.epw import EPW
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, ListedColormap, Normalize, to_hex
from matplotlib.patches import Rectangle
from pandas.tseries.frequencies import to_offset
from scipy.stats import weibull_min

from .categorical.categories import BEAUFORT_CATEGORIES
from .helpers import (
    OpenMeteoVariable,
    angle_from_north,
    angle_to_vector,
    cardinality,
    circular_weighted_mean,
    scrape_meteostat,
    scrape_openmeteo,
    wind_speed_at_height,
)
from .ladybug_extension.analysisperiod import (
    AnalysisPeriod,
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from .plot._timeseries import timeseries
from .plot.utilities import contrasting_color, format_polar_plot

# pylint: enable=E0401


@dataclass(init=True, eq=True, repr=True)
class Wind:
    """An object containing historic, time-indexed wind data.

    Args:
        wind_speeds (list[int | float | np.number]):
            An iterable of wind speeds in m/s.
        wind_directions (list[int | float | np.number]):
            An iterable of wind directions in degrees from North (with North at 0-degrees).
        datetimes (Union[pd.DatetimeIndex, list[Union[datetime, np.datetime64, pd.Timestamp]]]):
            An iterable of datetime-like objects.
        height_above_ground (float, optional):
            The height above ground (in m) where the input wind speeds and directions were
            collected. Defaults to 10m.
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

        if not len(self.wind_speeds) == len(self.wind_directions) == len(self.datetimes):
            raise ValueError("wind_speeds, wind_directions and datetimes must be the same length.")

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
    def from_dict(cls, d: dict) -> "Wind":
        """Create this object from a dictionary."""

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
    def from_json(cls, json_string: str) -> "Wind":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Convert this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w", encoding="utf-8") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "Wind":
        """Create this object from a JSON file."""
        with open(Path(path), "r", encoding="utf-8") as fp:
            return cls.from_json(fp.read())

    def to_csv(self, path: Path) -> Path:
        """Save this object as a csv file.

        Args:
            path (Path):
                The path containing the CSV file.

        Returns:
            Path:
                The resultant CSV file.
        """
        csv_path = Path(path)
        self.df.to_csv(csv_path)
        return csv_path

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        wind_speed_column: Any,
        wind_direction_column: Any,
        height_above_ground: float = 10,
        source: str = "DataFrame",
    ) -> "Wind":
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
                A source string to describe where the input data comes from.
                Defaults to "DataFrame"".
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df must be of type {type(pd.DataFrame)}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"The DataFrame's index must be of type {type(pd.DatetimeIndex)}")

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
    ) -> "Wind":
        """Create a Wind object from a csv containing wind speed and direction columns.

        Args:
            csv_path (Path):
                The path to the CSV file containing speed and direction columns,
                and a datetime index.
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
    def from_epw(cls, epw: Path | EPW) -> "Wind":
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
    ) -> "Wind":
        """Create a Wind object from data obtained from the Open-Meteo database
        of historic weather station data.

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
    ) -> "Wind":
        """Create a Wind object from data obtained from the Meteostat database
        of historic weather station data.

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
    def from_average(cls, wind_objects: list["Wind"], weights: list[float] = None) -> "Wind":
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
        wd_avg = np.array([circular_weighted_mean(i, weights) for _, i in df_wd.iterrows()])
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
        return pd.Series(self.wind_speeds, index=self.index, name="Wind Speed (m/s)").sort_index(
            ascending=True, inplace=False
        )

    @property
    def wd(self) -> pd.Series:
        """Convenience accessor for wind directions as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_directions, index=self.index, name="Wind Direction (degrees)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def df(self) -> pd.DataFrame:
        """Convenience accessor for wind direction and speed as a time-indexed
        pd.DataFrame object."""
        return pd.concat([self.wd, self.ws], axis=1)

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

    def mean_speed(self, remove_calm: bool = False) -> float:
        """Return the mean wind speed for this object.

        Args:
            remove_calm (bool, optional):
                Remove calm wind speeds before calculating the mean. Defaults to False.

        Returns:
            float:
                Mean wind speed.

        """
        return np.linalg.norm(self.filter_by_speed(min_speed=1e-10 if remove_calm else 0).mean_uv)

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

    def resample(self, rule: pd.DateOffset | pd.Timedelta | str) -> "Wind":
        """Resample the wind data collection to a different timestep.
        This can only be used to downsample.

        Args:
            rule (Union[pd.DateOffset, pd.Timedelta, str]):
                A rule for resampling. This uses the same inputs as a Pandas
                Series.resample() method.

        Returns:
            Wind:
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

        return Wind(
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
    ) -> "Wind":
        """Translate the object to a different height above ground.

        Args:
            target_height (float):
                Height to translate to (in m).
            terrain_roughness_length (float, optional):
                Terrain roughness (how big objects are to adjust translation).
                Defaults to 1.
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
        return Wind(
            wind_speeds=ws.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=target_height,
            source=f"{self.source} translated to {target_height}m",
        )

    def apply_directional_factors(self, directions: int, factors: tuple[float]) -> "Wind":
        """Adjust wind speed values by a set of factors per direction.
        Factors start at north, and move clockwise.

        Example:
            >>> wind = Wind.from_epw(epw_path)
            >>> wind.apply_directional_factors(
            ...     directions=4,
            ...     factors=(0.5, 0.75, 1, 0.75)
            ... )

        Where northern winds would be multiplied by 0.5, eastern winds by 0.75,
        southern winds by 1, and western winds by 0.75.

        Args:
            directions (int):
                The number of directions to bin wind-directions into.
            factors (tuple[float], optional):
                Adjustment factors per direction.

        Returns:
            Wind:
                An adjusted Wind object.
        """

        factors = np.array(factors).tolist()

        if len(factors) != directions:
            raise ValueError(
                f"number of factors ({len(factors)}) must equal number of directions ({directions})"
            )

        direction_binned = self.bin_data(directions=directions)
        directional_factor_lookup = dict(
            zip(*[np.roll(np.unique(direction_binned.iloc[:, 0]), 1), factors])
        )

        adjusted_wind_speed = self.ws * [
            directional_factor_lookup[i] for i in direction_binned.iloc[:, 0]
        ]

        return Wind(
            wind_speeds=adjusted_wind_speed.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
            source=f"{self.source} adjusted by factors {factors}",
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod,
    ) -> "Wind":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if analysis_period == AnalysisPeriod():
            return self

        if analysis_period.timestep != 1:
            raise ValueError("The timestep of the analysis period must be 1 hour.")

        # remove 29th Feb dates where present
        df = self.df
        df = df[~((df.index.month == 2) & (df.index.day == 29))]

        # filter available data
        possible_datetimes = [DateTime(dt.month, dt.day, dt.hour, dt.minute) for dt in df.index]
        lookup = dict(zip(AnalysisPeriod().datetimes, analysis_period_to_boolean(analysis_period)))
        mask = [lookup[i] for i in possible_datetimes]
        df = df[mask]

        if len(df) == 0:
            raise ValueError("No data remains within the given analysis_period filter.")

        return Wind.from_dataframe(
            df,
            wind_speed_column="Wind Speed (m/s)",
            wind_direction_column="Wind Direction (degrees)",
            height_above_ground=self.height_above_ground,
            source=f"{self.source} (filtered to {describe_analysis_period(analysis_period)})",
        )

    def filter_by_boolean_mask(self, mask: tuple[bool]) -> "Wind":
        """Filter the current object by a boolean mask.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if len(mask) != len(self.ws):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )

        if sum(mask) == len(self.ws):
            return self

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given boolean filters.")

        return Wind(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=f"{self.source} (filtered)",
        )

    def filter_by_time(
        self,
        months: tuple[float] = tuple(range(1, 13, 1)),  # type: ignore
        hours: tuple[int] = tuple(range(0, 24, 1)),  # type: ignore
        years: tuple[int] = tuple(range(1900, 2100, 1)),
    ) -> "Wind":
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

        mask = np.all(
            [
                self.datetimes.year.isin(years),
                self.datetimes.month.isin(months),
                self.datetimes.hour.isin(hours),
            ],
            axis=0,
        ).flatten()

        if mask.sum() == len(self.ws):
            return self

        if len(self.ws.iloc[mask]) == 0:
            raise ValueError("No data remains within the given time filters.")

        filtered_by = []
        if len(years) != len(range(1900, 2100, 1)):
            filtered_by.append("year")
        if len(months) != len(range(1, 13, 1)):
            filtered_by.append("month")
        if len(hours) != len(range(0, 24, 1)):
            filtered_by.append("hour")

        filtered_by = ", ".join(filtered_by)
        return Wind(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=f"{self.source} (filtered {filtered_by})",
        )

    def filter_by_direction(
        self, left_angle: float = 0, right_angle: float = 360, inclusive: bool = True
    ) -> "Wind":
        """Filter the current object by wind direction, based on the angle as
        observed from a location.

        Args:
            left_angle (float):
                The left-most angle, to the left of which wind speeds and
                directions will be removed.
            right_angle (float):
                The right-most angle, to the right of which wind speeds and
                directions will be removed.
            inclusive (bool, optional):
                Include values that are exactly the left or right angle values.

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

        return Wind(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=f"{self.source} filtered by direction ({left_angle}-{right_angle})",
        )

    def filter_by_speed(
        self, min_speed: float = 0, max_speed: float = 999, inclusive: bool = True
    ) -> "Wind":
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

        if min_speed == 0 and max_speed == 999:
            return self

        if inclusive:
            mask = (self.ws >= min_speed) & (self.ws <= max_speed)
        else:
            mask = (self.ws > min_speed) & (self.ws < max_speed)

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given speed filter.")

        return Wind(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
            source=f"{self.source} filtered by speed ({min_speed}-{max_speed})",
        )

    @staticmethod
    def _direction_bin_edges(directions: int) -> list[float]:
        """Calculate the bin edges for a given number of directions.

        Args:
            directions (int):
                The number of directions to calculate bin edges for.

        Returns:
            list[float]:
                A list of bin edges.
        """

        if directions <= 2:
            raise ValueError("directions must be > 2.")

        direction_bin_edges = np.unique(
            ((np.linspace(0, 360, directions + 1) - ((360 / directions) / 2)) % 360).tolist()
            + [0, 360]
        )
        return direction_bin_edges

    def process_direction_data(self, directions: int) -> tuple[pd.Series, list[tuple[float]]]:
        """Process wind direction data for this object.

        Args:
            directions (int):
                The number of directions to calculate bin edges for.

        Returns:
            tuple[pd.Series, list[tuple[float]]]:
                - A pd.Series containing the processed categorical data.
                - A list of tuples containing the bin edges.
        """

        # bin the wind directions
        categories = pd.cut(
            self.wd,
            bins=Wind._direction_bin_edges(directions),
            include_lowest=True,
        )

        x = [tuple([i.left, i.right]) for i in categories.cat.categories.tolist()]
        bin_tuples = x[1:-1]
        bin_tuples.append((bin_tuples[-1][1], bin_tuples[0][0]))

        mapper = dict(
            zip(
                *[
                    categories.cat.categories.tolist(),
                    [bin_tuples[-1]] + bin_tuples,
                ]
            )
        )

        # modify bin tuples and create mapping
        categories = pd.Series(
            [mapper[i] for i in categories],
            index=categories.index,
            name=categories.name,
        )

        return categories, bin_tuples

    def process_other_data(
        self,
        other_data: list[float] | pd.Series = None,
        other_bins: list[float] | int = None,
        name: str = None,
    ) -> tuple[pd.Series, list[tuple[float]]]:
        """Process aligned data for this object.

        Args:
            other_data (list[float] | pd.Series, optional):
                The other data to process.
                Defaults to None, which in turn uses Wind Speed.
            other_bins (list[float] | int, optional):
                The bins to sort this data into.
                Defaults to None, which if other_data is None, uses Beaufort scale bins.
            name (str, optional): A name to be given to the other data.
                Defaults to None which uses "other", or the name of the input Series
                if input is a Series.

        Returns:
            tuple[pd.Series, list[tuple[float]]]:
                - A pd.Series containing the processed categorical data.
                - A list of tuples containing the bin edges.
        """
        # prepare other data
        if other_data is None:
            other_data = self.ws
            if other_bins is None:
                other_bins = BEAUFORT_CATEGORIES.bins

        if len(other_data) != len(self):
            raise ValueError(f"other_data must be the same length as this {type(self)} object.")

        if isinstance(other_data, list | tuple | np.ndarray):
            other_data = pd.Series(
                other_data, index=self.index, name="other" if name is None else name
            )

        if isinstance(other_data, pd.Series):
            if not other_data.index.equals(self.index):
                raise ValueError(
                    f"other_data must have the same index as this {type(self)} object."
                )
            if name is not None:
                other_data.name = name

        # prepare other bins
        if other_bins is None:
            other_bins = 11

        if isinstance(other_bins, int):
            other_bins = np.linspace(min(other_data), max(other_data), 11)

        if len(other_bins) < 2:
            raise ValueError("other_bins must contain at least 2 values.")

        if min(other_data) < min(other_bins):
            raise ValueError(
                f"Data exists below the bounds of the given bins ({min(other_data)}<{min(other_bins)})."
            )
        if max(other_data) > max(other_bins):
            raise ValueError(
                f"Data exists above the bounds of the given bins ({max(other_data)}>{max(other_bins)})."
            )

        # bin the other data
        categories = pd.cut(other_data, bins=other_bins, include_lowest=False)

        bin_tuples = [tuple([i.left, i.right]) for i in categories.cat.categories.tolist()]

        mapper = dict(
            zip(
                *[
                    categories.cat.categories.tolist(),
                    bin_tuples,
                ]
            )
        )
        mapper[np.nan] = bin_tuples[0]

        # modify bin tuples and create mapping
        categories = pd.Series(
            [mapper[i] for i in categories],
            index=categories.index,
            name=categories.name,
        )

        return categories, bin_tuples

    def bin_data(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
    ) -> pd.DataFrame:
        """Create categories for wind direction and "other" data. By default "other"
        data is the wind speed associate with the wind direction in this object.

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.
            other_data (list[float], optional):
                An iterable of data to bin. Defaults to None.
            other_bins (list[float], optional):
                An iterable of bin edges to use for the other data. Defaults to None.

        Returns:
            pd.DataFrame:
                A DataFrame containing the wind direction categories and the "other"
                data categories.
        """

        other_categories, _ = self.process_other_data(other_data=other_data, other_bins=other_bins)
        direction_categories, _ = self.process_direction_data(directions=directions)

        return pd.concat([direction_categories, other_categories], axis=1)

    def histogram(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
        density: bool = False,
        remove_calm: bool = False,
    ) -> pd.DataFrame:
        """Bin data by direction, returning counts for each direction.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction.
                If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram.
                These bins are right inclusive.
            density (bool, optional):
                If True, then return the probability density function.
                Defaults to False.
            remove_calm (bool, optional):
                If True, then remove calm wind speeds from the histogram.
                Defaults to False.

        Returns:
            pd.DataFrame:
                A numpy array, containing the number or probability for each bin,
                for each direction bin.
        """

        other_categories, other_bin_tuples = self.process_other_data(
            other_data=other_data, other_bins=other_bins
        )
        direction_categories, direction_bin_tuples = self.process_direction_data(
            directions=directions
        )

        df = pd.concat([direction_categories, other_categories], axis=1)

        if remove_calm:
            df = df[self.ws > 1e-10]

        # pivot data
        df = (
            df.groupby([df.columns[0], df.columns[1]], observed=True)
            .value_counts()
            .unstack()
            .fillna(0)
            .astype(int)
        )

        # re-add missing rows/columns to the dataframe which were not in the cut data
        for b in other_bin_tuples:
            if b not in df.columns:
                df[b] = 0
        df.sort_index(axis=1, inplace=True)
        df = df.T
        for b in direction_bin_tuples:
            if b not in df.columns:
                df[b] = 0
        df.sort_index(axis=1, inplace=True)
        df = df.T

        if density:
            df = df / df.values.sum()

        return df

    def prevailing(
        self,
        directions: int = 36,
        n: int = 1,
        as_cardinal: bool = False,
        ignore_calm: bool = True,
        threshold: float = 1e-10,
    ) -> list[float] | list[str]:
        """Calculate the prevailing wind direction/s for this object.

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.
            n (int, optional):
                The number of prevailing directions to return. Default is 1.
            as_cardinal (bool, optional):
                If True, then return the prevailing directions as cardinal directions.
                Defaults to False.

        Returns:
            list[float] | list[str]:
                A list of wind directions.
        """

        binned = self.bin_data(directions=directions)

        if ignore_calm:
            binned = binned.loc[self.ws > threshold]

        prevailing_angles = binned.iloc[:, 0].value_counts().index[:n]

        if as_cardinal:
            card = []
            for i in prevailing_angles:
                if i[0] < i[1]:
                    card.append(cardinality(np.mean(i), directions=32))
                else:
                    card.append(cardinality(0, directions=32))
            return card

        return prevailing_angles.tolist()

    def probabilities(
        self,
        directions: int = 36,
        percentiles: tuple[float, float] = (0.5, 0.95),
        other_data: list[float] = None,
    ) -> pd.DataFrame:
        """Calculate the probabilities of wind speeds at the given percentiles,
        for the direction bins specified.

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.
            percentiles (tuple[float, float], optional):
                A tuple of floats between 0-1 describing percentiles.
                Defaults to (0.5, 0.95).
            other_data (list[float], optional):
                A list of other data to bin by direction.
                If None, then wind speed will be used.

        Returns:
            pd.DataFrame:
                A table showing probabilities at the given percentiles.
        """

        df = self.bin_data(directions=directions, other_data=other_data)

        if other_data is None:
            data_name = self.ws.name
            s = pd.Series(self.ws.values, name=data_name, index=self.index)
        else:
            data_name = df.iloc[:, 1].name
            s = pd.Series(other_data, name=data_name, index=self.index)

        # group data
        temp = s.groupby(df.iloc[:, 0], observed=True).quantile(percentiles).unstack()

        # rename columns to be percentages - potentially unecessary but makes it clearer for users
        temp.columns = [f"{i:0.1%}" for i in temp.columns]
        temp.columns.name = s.name

        return temp

    def _probability_density_function(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
    ) -> pd.DataFrame:
        """Create a table with the probability density function for each
        "other_data" and direction.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.

        Returns:
            pd.DataFrame:
                A probability density table.
        """

        hist = self.histogram(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
            density=False,
            remove_calm=True,
        )
        return pd.DataFrame(
            hist.values.T / hist.sum(axis=1).values,
            index=hist.columns,
            columns=hist.index,
        )

    def pdf(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
    ) -> pd.DataFrame:
        """Alias for the probability_density_function method."""
        return self._probability_density_function(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
        )

    def _cumulative_density_function(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
    ) -> pd.DataFrame:
        """Create a table with the cumulative probability density function for each
        "other_data" and direction.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.

        Returns:
            pd.DataFrame:
                A cumulative density table.
        """

        return self._probability_density_function(
            directions=directions, other_data=other_data, other_bins=other_bins
        ).cumsum(axis=0)

    def cdf(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
    ) -> pd.DataFrame:
        """Alias for the cumulative_density_function method."""
        return self._cumulative_density_function(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
        )

    def exceedance(
        self,
        limit_value: float,
        directions: int = 36,
        other_data: list[float] = None,
    ) -> pd.DataFrame:
        """Calculate the % of time where a limit-value is exceeded for each direction.

        Args:
            limit_value (float):
                The value above which speed frequency will be be calculated.
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.

        Returns:
            pd.DataFrame:
                A table containing exceedance values.
        """

        other_categories, _ = self.process_other_data(other_data=other_data)
        direction_categories, direction_bin_tuples = self.process_direction_data(
            directions=directions
        )

        if other_data is None:
            other_data = self.ws

        exceedance = []
        for month in range(1, 13, 1):
            mask = self.index.month == month
            meets = other_data[mask] > limit_value
            temp = pd.concat([direction_categories[mask], meets], axis=1)
            exceedance.append(
                (temp.iloc[:, 1].groupby([temp.iloc[:, 0]]).sum() / mask.sum()).rename(
                    calendar.month_abbr[month]
                )
            )

        df = pd.concat(exceedance, axis=1)
        df.columns.name = f"{other_categories.name} > {limit_value}"

        # re-add missing rows if they were not in the cut data
        for db in direction_bin_tuples:
            if db not in df.index:
                df.loc[db] = 0
        df.sort_index(axis=0, inplace=True)

        return df.fillna(0)

    def wind_matrix(self, other_data: pd.Series = None) -> pd.DataFrame:
        """Calculate average wind direction and speed (or aligned other data)
        for each month and hour of in the Wind object.

        Args:
            other_data (pd.Series, optional):
                The other data to calculate the matrix for.

        Returns:
            pd.DataFrame:
                A DataFrame containing average other_data and direction for each
                month and hour of day.
        """

        if other_data is None:
            other_data = self.ws

        if not isinstance(other_data, pd.Series):
            raise ValueError("other_data must be a time indexed pandas Series.")

        if len(other_data) != len(self.wd):
            raise ValueError(f"other_data must be the same length as this {type(self)} object.")

        if not all(other_data.index == self.wd.index):
            raise ValueError("other_data must have the same index as this Wind object.")

        wind_directions = (
            (
                (
                    self.wd.groupby([self.wd.index.month, self.wd.index.hour], axis=0).apply(
                        circular_weighted_mean
                    )
                )
                % 360
            )
            .unstack()
            .T
        )
        wind_directions.columns = [calendar.month_abbr[i] for i in wind_directions.columns]
        _other_data = (
            other_data.groupby([other_data.index.month, other_data.index.hour], axis=0)
            .mean()
            .unstack()
            .T
        )
        _other_data.columns = [calendar.month_abbr[i] for i in _other_data.columns]

        df = pd.concat([wind_directions, _other_data], axis=1, keys=["direction", "other"])
        df.index.name = "hour"

        return df

    def summarise(self, directions: int = 36) -> list[str]:
        """Generate a textual summary of the current object.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.

        Returns:
            list[str]:
                A list of strings describing the current object.
        """

        binned = self.bin_data(directions=directions)
        prevailing_angles = binned.iloc[:, 0].value_counts()

        start_time = self.datetimes[0]
        end_time = self.datetimes[-1]

        prevailing_angle = prevailing_angles.index[0]
        prevailing_count = prevailing_angles.iloc[0]
        prevailing_cardinal = (
            cardinality(np.mean(prevailing_angle), directions=32)
            if prevailing_angle[0] < prevailing_angle[1]
            else cardinality(0, directions=32)
        )
        prevailing_mean = self.ws.values[binned.iloc[:, 0] == prevailing_angle].mean()
        prevailing_max = self.ws.values[binned.iloc[:, 0] == prevailing_angle].max()

        # pylint: disable=line-too-long
        return_strings = []
        from_epw = self.source.endswith(".epw")
        if from_epw:
            return_strings.append(
                f'This summary describes wind speed and direction relationship at {self.height_above_ground}m above ground for the file "{self.source}".'
            )
        else:
            return_strings.append(
                f"This summary describes historic wind speed and direction relationship for the period {start_time} to {end_time} at {self.height_above_ground}m above ground, from {self.source}."
            )

        return_strings.append(
            f"The prevailing wind direction is between {prevailing_angle[0]}° and {prevailing_angle[1]}° (or {prevailing_cardinal}), accounting for {prevailing_count} of {len(self.wd)} timesteps."
        )
        return_strings.append(
            f"Prevailing wind average speed is {prevailing_mean:.2f}m/s, with a maximum of {prevailing_max:.2f}m/s from that direction."
        )

        if from_epw:
            return_strings.append(
                f"Peak wind speeds were observed during {self.ws.idxmax():%B} at {self.ws.idxmax():%H:%M}, reaching {self.ws.max():.2f}m/s from {self.wd.loc[self.ws.idxmax()]}°.",
            )
        else:
            return_strings.append(
                f"Peak wind speeds were observed at {self.ws.idxmax()}, reaching {self.ws.max():.2f}m/s from {self.wd.loc[self.ws.idxmax()]}°.",
            )

        return_strings.append(f"{self.calm():.2%} of the time, wind speeds are calm (≤ 1e-10m/s).")
        return return_strings
        # pylint: enable=line-too-long

    def prevailing_wind_speeds(
        self, n: int = 1, directions: int = 36, ignore_calm: bool = True, threshold: float = 1e-10
    ) -> tuple[list[pd.Series], list[tuple[float, float]]]:
        """Gets the wind speeds for the prevailing directions

        Args:
            n (int):
                Number of prevailing directions to return. Defaults to 1
            directions (int):
                Number of direction bins to use when calculating the prevailing directions.
                Defaults to 36
            ignore_calm (bool):
                Whether to ignore calm hours when getting the prevailing directions.
                Defaults to True
            threshold (float):
                The threshold for calm hours. Defaults to 1e-10

        Returns:
            (list[pandas.Series], list[(float, float)]):
                Tuple containing a list of time-indexed series containing wind
                speed data for each prevailing direction, from most to least prevailing,
                and a list of wind directions corresponding to the serieses.
        """

        prevailing_directions = self.prevailing(
            n=n, directions=directions, ignore_calm=ignore_calm, threshold=threshold
        )

        prevailing_wind_speeds = [
            self.ws.loc[
                self.bin_data(directions=directions)["Wind Direction (degrees)"] == direction
            ]
            for direction in prevailing_directions
        ]

        return (prevailing_wind_speeds, prevailing_directions)

    def weibull_pdf(self) -> tuple[float]:
        """Calculate the parameters of an exponentiated Weibull continuous random variable.

        Returns:
            k (float):
                Shape parameter
            loc (float):
                Location parameter.
            c (float):
                Scale parameter.
        """

        ws = self.ws.values
        ws = ws[ws != 0]
        ws = ws[~np.isnan(ws)]

        try:
            return weibull_min.fit(ws.tolist())
        except ValueError as exc:
            warnings.warn(f"Not enough data to calculate Weibull parameters.\n{exc}")
            return (np.nan, np.nan, np.nan)  # type: ignore

    def weibull_directional(self, directions: int = 36) -> pd.DataFrame:
        """Calculate directional weibull coefficients for the given number of directions.

        Args:
            directions (int, optional):
                The number of directions to use. Defaults to 36.

        Returns:
            pd.DataFrame:
                A DataFrame object.
        """

        hist = self.histogram(directions=directions)
        d = {}
        for left, right in hist.index:
            d[(left, right)] = self.filter_by_direction(
                left_angle=left, right_angle=right
            ).weibull_pdf()
        df = pd.DataFrame(d, index=["k", "loc", "c"])
        df.columns = df.columns.tolist()
        return df.T

    ##################################
    # PLOTTING/VISUALISATION METHODS #
    ##################################

    def plot_timeseries(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:  # type: ignore
        """Create a simple line plot of wind speed.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. If None, the current axes will be used.
            **kwargs:
                Additional keyword arguments to pass to the function. These include:
                title (str, optional):
                    A title for the plot. Defaults to None.

        Returns:
            plt.Axes:
                A matplotlib Axes object.

        """

        if ax is None:
            ax = plt.gca()

        ax.set_title(textwrap.fill(f"{self.source}", 75))

        timeseries(self.ws, ax=ax, **kwargs)

        ax.set_ylabel(self.ws.name)

        return ax

    def plot_windmatrix(
        self,
        ax: plt.Axes = None,
        show_values: bool = False,
        show_arrows: bool = True,
        other_data: pd.Series = None,
        **kwargs,
    ) -> plt.Axes:
        """Create a plot showing the annual wind speed and direction bins
        using the month_time_average method.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. If None, the current axes will be used.
            show_values (bool, optional):
                Whether to show values in the cells. Defaults to False.
            show_arrows (bool, optional):
                Whether to show the directional arrows on each patch.
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

        if other_data is None:
            other_data = self.ws

        title = self.source
        nt = kwargs.pop("title", None)
        if nt is not None:
            title += f"\n{nt}"
        ax.set_title(textwrap.fill(f"{title}", 75))

        df = self.wind_matrix(other_data=other_data)
        _other_data = df["other"]
        _wind_directions = df["direction"]

        if any(
            [
                _other_data.shape != (24, 12),
                _wind_directions.shape != (24, 12),
                _wind_directions.shape != _other_data.shape,
                not _wind_directions.index.equals(_other_data.index),
                not _wind_directions.columns.equals(_other_data.columns),
                # not np.array_equal(_wind_directions.index, _other_data.index),
                # not np.array_equal(_wind_directions.columns, _other_data.columns),
            ]
        ):
            raise ValueError(
                "The other_data and wind_directions must cover all months of the "
                "year, and all hours of the day, and align with each other."
            )

        cmap = kwargs.pop("cmap", "YlGnBu")
        vmin = kwargs.pop("vmin", _other_data.values.min())
        vmax = kwargs.pop("vmax", _other_data.values.max())
        cbar_title = kwargs.pop("cbar_title", None)
        unit = kwargs.pop("unit", None)
        norm = kwargs.pop("norm", Normalize(vmin=vmin, vmax=vmax, clip=True))
        mapper = kwargs.pop("mapper", ScalarMappable(norm=norm, cmap=cmap))

        pc = ax.pcolor(_other_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        _x = -np.sin(np.deg2rad(_wind_directions.values))
        _y = -np.cos(np.deg2rad(_wind_directions.values))
        direction_matrix = angle_from_north([_x, _y])
        if show_arrows:
            arrow_scale = 0.8
            ax.quiver(
                np.arange(1, 13, 1) - 0.5,
                np.arange(0, 24, 1) + 0.5,
                (_x * _other_data.values / 2) * arrow_scale,
                (_y * _other_data.values / 2) * arrow_scale,
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
                        f"{direction_matrix[_yy][_xx]:0.0f}°",
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

        ax.set_xticks(np.arange(1, 13, 1) - 0.5)
        ax.set_xticklabels([calendar.month_abbr[i] for i in np.arange(1, 13, 1)])
        ax.set_yticks(np.arange(0, 24, 1) + 0.5)
        ax.set_yticklabels([f"{i:02d}:00" for i in np.arange(0, 24, 1)])
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)

        cb = plt.colorbar(pc, label=cbar_title, pad=0.01)
        cb.outline.set_visible(False)

        return ax

    def plot_densityfunction(
        self,
        ax: plt.Axes = None,
        speed_bins: list[float] | int = 11,
        percentiles: tuple[float] = (0.5, 0.95),
        function: str = "pdf",
        ylim: tuple[float] = None,
    ) -> plt.Axes:
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

        if function not in ["pdf", "cdf"]:
            raise ValueError('function must be either "pdf" or "cdf".')

        if ax is None:
            ax = plt.gca()

        ax.set_title(
            f"{str(self)}\n{'Probability Density Function' if function == 'pdf' else 'Cumulative Density Function'}"
        )

        self.ws.plot.hist(
            ax=ax,
            density=True,
            bins=speed_bins,
            cumulative=True if function == "cdf" else False,
        )

        for percentile in percentiles:
            x = np.quantile(self.ws, percentile)
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

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.25)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))

        return ax

    def plot_windrose(
        self,
        ax: plt.Axes = None,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
        colors: list[str | tuple[float] | Colormap] = None,
        title: str = None,
        legend: bool = True,
        ylim: tuple[float] = None,
        label: bool = False,
    ) -> plt.Axes:
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
            colors: (str | tuple[float] | Colormap, optional):
                A list of colors to use for the other bins. May also be a colormap.
                Defaults to the colors used for Beaufort wind comfort categories.
            title (str, optional):
                title to display above the plot. Defaults to the source of this wind object.
            legend (bool, optional):
                Set to False to remove the legend. Defaults to True.
            ylim (tuple[float], optional):
                The y-axis limits. Defaults to None.
            label (bool, optional):
                Set to False to remove the bin labels. Defaults to False.

        Returns:
            plt.Axes: The axes object.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        # create grouped data for plotting
        binned = self.histogram(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
            density=True,
            remove_calm=True,
        )

        # set colors
        if colors is None:
            if other_data is None:
                colors = [
                    to_hex(BEAUFORT_CATEGORIES.cmap(i))
                    for i in np.linspace(0, 1, len(binned.columns))
                ]
            else:
                colors = [
                    to_hex(plt.get_cmap("viridis")(i))
                    for i in np.linspace(0, 1, len(binned.columns))
                ]
        if isinstance(colors, str):
            colors = plt.get_cmap(colors)
        if isinstance(colors, Colormap):
            colors = [to_hex(colors(i)) for i in np.linspace(0, 1, len(binned.columns))]
        if isinstance(colors, list | tuple):
            if len(colors) != len(binned.columns):
                raise ValueError(
                    f"colors must be a list of length {len(binned.columns)}, or a colormap."
                )

        # HACK to ensure that bar ends are curved when using a polar plot.
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        hist_ax = plt.Axes(fig, rect)
        hist_ax.bar(np.array([1]), np.array([1]))

        if title is None or title == "":
            ax.set_title(textwrap.fill(f"{self.source}", 75))
        else:
            ax.set_title(title)

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
            if label:
                ax.text(x, y, f"{y:0.1%}", ha="center", va="center", fontsize="x-small")
            x += theta_width
        local_cmap = ListedColormap(np.array(color_list).flatten())
        pc = PatchCollection(patches, cmap=local_cmap)
        pc.set_array(np.arange(len(color_list)))
        ax.add_collection(pc)

        # construct legend
        if legend:
            handles = [
                mpatches.Patch(color=colors[n], label=f"{i} to {j}")
                for n, (i, j) in enumerate(binned.columns.values)
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
        if ylim is None:
            ylim = (0, max(binned.sum(axis=1)))
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple of length 2.")
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        format_polar_plot(ax, yticklabels=True)

        return ax

    def plot_windhistogram(
        self,
        ax: plt.Axes = None,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = None,
        density: bool = False,
        cmap: str | Colormap = "YlGnBu",
        show_values: bool = True,
        vmin: float = None,
        vmax: float = None,
    ) -> plt.Axes:
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
            cmap (str | Colormap, optional):
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
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
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
            cb.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))
        cb.outline.set_visible(False)

        ax.set_title(textwrap.fill(f"{self.source}", 75))

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

    # TODO - add WindProfile plot here

    # TODO - add Climate Consultant-style "wind wheel" plot here
    # (http://2.bp.blogspot.com/-F27rpZL4VSs/VngYxXsYaTI/AAAAAAAACAc/yoGXmk13uf8/s1600/CC-graphics%2B-%2BWind%2BWheel.jpg)

    # TODO - add radial version of plot_windhistogram
