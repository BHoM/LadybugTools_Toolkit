"""Methods for working with time-indexed wind data."""

# pylint: disable=E0401
import calendar
import warnings
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# pylint: enable=E0401

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

import pandas as pd

from ladybug.dt import DateTime
from ladybug.epw import EPW

from tqdm import tqdm

from .bhom.analytics import bhom_analytics


from .plot.utilities import contrasting_color, format_polar_plot
from .categorical.categories import BEAUFORT_CATEGORIES
from .helpers import (
    angle_from_north,
    angle_to_vector,
    OpenMeteoVariable,
    circular_weighted_mean,
    convert_keys_to_snake_case,
    rolling_window,
    scrape_meteostat,
    scrape_openmeteo,
    weibull_directional,
    weibull_pdf,
    wind_direction_average,
    wind_speed_at_height,
)
from .ladybug_extension.analysisperiod import (
    AnalysisPeriod,
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
)
from .plot._timeseries import timeseries
from .directionbins import DirectionBins


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
            The height above ground (in m) where the input wind speeds and directions were collected. Defaults to 10m.
    """

    wind_speeds: list[float]
    wind_directions: list[float]
    datetimes: list[datetime] | pd.DatetimeIndex
    height_above_ground: Optional[float]

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

        for ws in self.wind_speeds:
            if np.isnan(ws):
                raise ValueError("wind_speeds contains null values.")
            if ws < 0:
                raise ValueError("wind_speeds must be >= 0")

        for n, wd in enumerate(self.wind_directions):
            if np.isnan(wd):
                raise ValueError("wind_directions contains null values.")
            if wd < 0 or wd > 360:
                raise ValueError("wind_directions must be within 0-360")
            if wd == 360:
                self.wind_directions[n] = 0

    def __len__(self) -> int:
        return len(self.datetimes)

    def __repr__(self) -> str:
        """The printable representation of the given object"""
        return (
            f"{self.__class__.__name__}({min(self.datetimes):%Y-%m-%d} to "
            f"{max(self.datetimes):%Y-%m-%d}, n={len(self.datetimes)} @{self.freq}, "
            f"@{self.height_above_ground}m)"
        )

    def to_dict(self) -> dict:
        """Return the object as a dictionary."""

        return {
            "_t": "BH.oM.LadybugTools.Wind",
            "WindSpeeds": [float(i) for i in self.wind_speeds],
            "WindDirections": [float(i) for i in self.wind_directions],
            "Datetimes": [i.isoformat() for i in self.datetimes],
            "HeightAboveGround": self.height_above_ground,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Wind":
        """Create a DirectionBins object from a dictionary."""

        d = convert_keys_to_snake_case(d)

        # datetimes = []
        # for dt in d["datetimes"]:
        #     if isinstance(dt, (str)):
        #         datetimes.append(datetime.fromordinal(int(dt)))

        return cls(
            wind_speeds=d["wind_speeds"],
            wind_directions=d["wind_directions"],
            datetimes=pd.to_datetime(d["datetimes"]),
            height_above_ground=d["height_above_ground"],
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

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "Wind":
        """Create this object from a JSON file."""
        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

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

    #################
    # CLASS METHODS #
    #################

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        wind_speed_column: Any,
        wind_direction_column: Any,
        height_above_ground: float = 10,
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

        return Wind(
            wind_speeds=df[wind_speed_column].tolist(),
            wind_directions=df[wind_direction_column].tolist(),
            datetimes=df.index.tolist(),
            height_above_ground=height_above_ground,
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
        df = pd.read_csv(csv_path, **kwargs)
        return Wind.from_dataframe(
            df,
            wind_speed_column=wind_speed_column,
            wind_direction_column=wind_direction_column,
            height_above_ground=height_above_ground,
        )

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "Wind":
        """Create a Wind object from an EPW file or object.

        Args:
            epw (Path | EPW):
                The path to the EPW file, or an EPW object.
        """

        if isinstance(epw, (str, Path)):
            epw = EPW(epw)

        return Wind(
            wind_speeds=epw.wind_speed.values,
            wind_directions=epw.wind_direction.values,
            datetimes=analysis_period_to_datetimes(AnalysisPeriod()),
            height_above_ground=10,
        )

    @classmethod
    def from_openmeteo(
        cls,
        latitude: float,
        longitude: float,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> "Wind":
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
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        df = scrape_openmeteo(
            latitude,
            longitude,
            start_date,
            end_date,
            [OpenMeteoVariable.WINDSPEED_10M, OpenMeteoVariable.WINDDIRECTION_10M],
        )
        df.dropna(how="any", axis=0, inplace=True)
        wind_speeds = np.multiply(
            df["windspeed_10m (km/h)"].astype(float), 0.277778
        ).tolist()
        wind_directions = df["winddirection_10m (°)"].astype(float).tolist()
        if len(wind_speeds) == 0 or len(wind_directions) == 0:
            raise ValueError(
                "OpenMeteo did not return any data for the given latitude, longitude and start/end dates."
            )
        datetimes = df.index.tolist()
        return Wind(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            datetimes=datetimes,
            height_above_ground=10,
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
            altitude (float, optional):
                The altitude of the target site, in meters. Defaults to 10.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        df = scrape_meteostat(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            altitude=altitude,
            convert_units=True,
        )[["Wind Speed (m/s)", "Wind Direction (degrees)"]]

        df.dropna(axis=0, inplace=True, how="any")

        return Wind.from_dataframe(
            df=df,
            wind_speed_column="Wind Speed (m/s)",
            wind_direction_column="Wind Direction (degrees)",
            height_above_ground=altitude,
        )

    @classmethod
    def from_average(
        cls, wind_objects: list["Wind"], weights: list[float] = None
    ) -> "Wind":
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""

        # create default weightings if None
        if weights is None:
            weights = [1 / len(wind_objects)] * len(wind_objects)
        else:
            if sum(weights) != 1:
                raise ValueError("weights must total 1.")

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
        return Wind(
            wind_speeds=ws_avg.tolist(),
            wind_directions=wd_avg.tolist(),
            datetimes=dts,
            height_above_ground=np.average(
                [i.height_above_ground for i in wind_objects], weights=weights
            ),
        )

    @classmethod
    def from_uv(
        cls,
        u: list[float],
        v: list[float],
        datetimes: list[datetime],
        height_above_ground: float = 10,
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
                The height above ground (in m) where the input wind speeds and directions were collected.
                Defaults to 10m.

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

        return Wind(
            wind_speeds=wind_speed.tolist(),
            wind_directions=wind_direction.tolist(),
            datetimes=datetimes,
            height_above_ground=height_above_ground,
        )

    ###################
    # GENERAL METHODS #
    ###################

    @bhom_analytics()
    def mean(self) -> float:
        """Return the mean wind speed for this object."""
        return self.ws.mean()

    @bhom_analytics()
    def min(self) -> float:
        """Return the min wind speed for this object."""
        return self.ws.min()

    @bhom_analytics()
    def max(self) -> float:
        """Return the max wind speed for this object."""
        return self.ws.max()

    @bhom_analytics()
    def median(self) -> float:
        """Return the median wind speed for this object."""
        return self.ws.median()

    @bhom_analytics()
    def calm(self, threshold: float = 0.1) -> float:
        """Return the proportion of timesteps "calm" (i.e. wind-speed ≤ 0.1)."""
        return (self.ws <= threshold).sum() / len(self.ws)

    @bhom_analytics()
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

    @bhom_analytics()
    def resample(self, rule: pd.DateOffset | pd.Timedelta | str) -> "Wind":
        """Resample the wind data collection to a different timestep. If upsampling then 0m/s
            and prevailing winds will be added to the data. If downsampling, then the average
            speed and direction will be used.

        Args:
            rule (Union[pd.DateOffset, pd.Timedelta, str]):
                A rule for resampling. This uses the same inputs as a Pandas Series.resample() method.

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

        resampled_speeds = self.ws.resample(rule).mean()
        resampled_datetimes = resampled_speeds.index.tolist()
        resampled_directions = self.wd.resample(rule).apply(wind_direction_average)

        if resampled_speeds.isna().sum() > 0:
            warnings.warn(
                (
                    'Gaps in the source data mean that resampling must introduce "0m/s" wind speeds in '
                    'order to create a valid dataset. This artificially increases the amount of "calm" hours.'
                )
            )
            resampled_speeds.fillna(0, inplace=True)

        if resampled_directions.isna().sum() > 0:
            prevailing_direction = self.prevailing(
                DirectionBins(32), n=1, as_angle=True
            )[0]
            warnings.warn(
                (
                    "Gaps in the source data mean that resampling must introduce artifical wind directions "
                    "in order to create a valid dataset. The wind direction used is the average of the whole "
                    "input dataset. This artificially increases the amount of wind from the prevailing "
                    "direction."
                )
            )
            resampled_directions.fillna(prevailing_direction, inplace=True)

        return Wind(
            wind_speeds=resampled_speeds.tolist(),
            wind_directions=resampled_directions.tolist(),
            datetimes=resampled_datetimes,
            height_above_ground=self.height_above_ground,
        )

    @bhom_analytics()
    def prevailing(
        self,
        direction_bins: DirectionBins = DirectionBins(),
        n: int = 1,
        as_angle: bool = False,
    ) -> list[float] | list[str]:
        """Calculate the prevailing wind direction/s for this object.

        Args:
            direction_bins (DirectionBins, optional):
                The number of wind directions to bin values into.
            n (int, optional):
                The number of prevailing directions to return. Default is 1.
            as_angle (bool, optional):
                Return directions as an angle or as a cardinal direction. Default is False to return cardinal direction.

        Returns:
            list[float] | list[str]:
                A list of wind directions.
        """

        return direction_bins.prevailing(self.wd.tolist(), n, as_angle)

    @bhom_analytics()
    def average_direction(self) -> tuple[float, float]:
        """Calculate the average speed and direction for this object.

        Returns:
            tuple[float, float]:
                A tuple containing the average speed and direction.
        """
        return angle_from_north(self.uv.values.mean(axis=0))

    @bhom_analytics()
    def probabilities(
        self,
        direction_bins: DirectionBins = DirectionBins(),
        percentiles: tuple[float, float] = (0.5, 0.95),
    ) -> pd.DataFrame:
        """Calculate the probabilities of wind speeds at the given percentiles, for the direction bins specified.

        Args:
            direction_bins (DirectionBins, optional):
                A DirectionBins object. Defaults to DirectionBins().
            percentiles (tuple[float, float], optional):
                A tuple of floats between 0-1 describing percentiles. Defaults to (0.5, 0.95).

        Returns:
            pd.DataFrame:
                A table showing probabilities at the given percentiles.
        """

        binned_data = direction_bins.bin_data(self.wd.tolist(), self.ws.tolist())

        probabilities = []
        for percentile in percentiles:
            temp = []
            for _, vals in binned_data.items():
                temp.append(np.quantile(vals, percentile))
            probabilities.append(temp)

        df = pd.DataFrame(
            data=probabilities, columns=direction_bins.midpoints, index=percentiles  # type: ignore
        )
        return df.T

    @bhom_analytics()
    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod | tuple[AnalysisPeriod],
    ) -> "Wind":
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

        return Wind.from_dataframe(
            df,
            wind_speed_column="Wind Speed (m/s)",
            wind_direction_column="Wind Direction (degrees)",
            height_above_ground=self.height_above_ground,
        )

    @bhom_analytics()
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

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given boolean filters.")

        return Wind(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
        )

    @bhom_analytics()
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

        return Wind(
            wind_speeds=self.ws.iloc[indices].tolist(),
            wind_directions=self.wd.iloc[indices].tolist(),
            datetimes=self.datetimes[indices],
            height_above_ground=self.height_above_ground,
        )

    @bhom_analytics()
    def filter_by_direction(
        self, left_angle: float = 0, right_angle: float = 360, inclusive: bool = True
    ) -> "Wind":
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

        return Wind(
            wind_speeds=self.ws.values[mask].tolist(),
            wind_directions=self.wd.values[mask].tolist(),
            datetimes=self.datetimes[mask],
            height_above_ground=self.height_above_ground,
        )

    @bhom_analytics()
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
        )

    @bhom_analytics()
    def frequency_table(
        self,
        speed_bins: tuple[float] = None,
        direction_bins: DirectionBins = DirectionBins(),
        density: bool = False,
        include_counts: bool = False,
    ) -> pd.DataFrame:
        """Create a table with wind direction per rows, and wind speed per column.

        Args:
            speed_bins (list[float], optional):
                A list of bins edges, between which wind speeds will be binned.
                Defaults to (0, 0.1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 99.9).
            direction_bins (DirectionBins, optional):
                A DirectionBins object. Defaults to DirectionBins().
            density (bool, optional):
                Set to True to return density (0-1) instead of count. Defaults to False.
            include_counts (bool, optional):
                Include counts for each row and columns if True. Defaults to False.

        Returns:
            pd.DataFrame:
                A frequency table.
        """

        if speed_bins is None:
            speed_bins = np.linspace(0, self.ws.max() + 1, 21)

        if direction_bins.directions <= 3:
            raise ValueError(
                "At least 4 directions should be specified in the DirectionBins object."
            )

        # bin data using speed bins and direction
        ds = []
        speed_bin_labels = []
        for low, high in rolling_window(speed_bins, 2):
            speed_bin_labels.append((low, high))
            mask = (self.ws >= low) & (self.ws < high)
            binned = direction_bins.bin_data(self.wd.values[mask])
            ds.append({k: len(v) for k, v in binned.items()})
        df = pd.DataFrame.from_dict(ds).T
        df.columns = speed_bin_labels

        if density:
            df = df / df.sum().sum()

        if include_counts:
            df = pd.concat([df, df.sum(axis=1).rename("sum")], axis=1)
            df = pd.concat(
                [df, df.sum(axis=0).to_frame().T.rename(index={0: "sum"})], axis=0
            )

        return df.T

    @bhom_analytics()
    def cumulative_density_function(
        self,
        speed_bins: tuple[float] = None,
        direction_bins: DirectionBins = DirectionBins(),
    ) -> pd.DataFrame:
        """Create a table with the cumulative probability density function for each speed and direction.

        Args:
            speed_bins (list[float], optional):
                A list of bins edges, between which wind speeds will be binned.
                Defaults to (0, 0.1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 99.9).
            direction_bins (DirectionBins, optional):
                A DirectionBins object. Defaults to DirectionBins().

        Returns:
            pd.DataFrame:
                A cumulative density table.
        """
        return self.frequency_table(
            speed_bins=speed_bins, density=True, direction_bins=direction_bins
        ).cumsum(axis=0)

    @bhom_analytics()
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
        return weibull_pdf(self.ws.tolist())

    @bhom_analytics()
    def weibull_directional(
        self, direction_bins: DirectionBins = DirectionBins()
    ) -> pd.DataFrame:
        """Calculate directional weibull coefficients for the given number of directions.

        Args:
            direction_bins (DirectionBins, optional):
                A DirectionBins object. Defaults to DirectionBins().

        Returns:
            pd.DataFrame:
                A DataFrame object.
        """
        binned_data = direction_bins.bin_data(self.wd.tolist(), self.ws.tolist())
        return weibull_directional(binned_data)

    @bhom_analytics()
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
        return Wind(
            wind_speeds=ws.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=target_height,
        )

    @bhom_analytics()
    def apply_directional_factors(
        self, direction_bins: DirectionBins, factors: tuple[float]
    ) -> "Wind":
        """Adjust wind speed values by a set of factors per direction.

        Args:
            direction_bins (DirectionBins):
                The number of directions to bin wind-directions into.
            factors (tuple[float], optional):
                Adjustment factors per direction. Defaults to (i for i in range(8)).

        Returns:
            Wind:
                An adjusted Wind object.
        """

        factors = np.array(factors).tolist()

        if len(factors) != len(direction_bins):
            raise ValueError(
                f"number of factors must equal number of directional bins ({len(direction_bins)} != {len(factors)})"
            )

        # repeat first element in inputs to apply to the two directions centered about North
        if direction_bins.is_split:
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

        return Wind(
            wind_speeds=ws.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
        )

    @bhom_analytics()
    def exceedance(
        self,
        limit_value: float,
        direction_bins: DirectionBins = DirectionBins(),
        agg: str = "max",
    ) -> pd.DataFrame:
        """Calculate the % of time where a limit-value is exceeded for the given direction bins.

        Args:
            limit_value (float):
                The value above which speed frequency will be be calculated.
            direction_bins (DirectionBins, optional):
                A DirectionBins object defining the bins into which wind will be split according to direction.
            agg (str, optional):
                The aggregation method to use. Defaults to "max".

        Returns:
            pd.DataFrame:
                A table containing exceedance values.
        """

        with tqdm(range(1, 13, 1)) as t:
            exceedances = []
            for month in t:
                t.set_description(
                    f"Calculating wind speed exceedance > {limit_value}m/s for {calendar.month_abbr[month]}"
                )

                try:
                    _w = self.filter_by_time(months=[month])
                except ValueError:
                    exceedances.append([0] * len(direction_bins))
                    continue
                binned_data = direction_bins.bin_data(_w.wd, _w.ws)
                temp_exceedances = []
                for _, values in binned_data.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _exceedance_value = (
                            np.array(values) > limit_value
                        ).sum() / len(values)
                    temp_exceedances.append(_exceedance_value)
                exceedances.append(temp_exceedances)

        df = pd.DataFrame(exceedances).fillna(0)

        df.index = [calendar.month_abbr[i + 1] for i in df.index]  # type: ignore
        df.columns = [(i[0], i[1]) for i in binned_data.keys()]  # type: ignore

        # add aggregation
        df = pd.concat([df, df.agg(agg, axis=1).rename(agg)], axis=1)
        df = pd.concat(
            [df, df.agg(agg, axis=0).to_frame().T.rename(index={0: agg})], axis=0
        )

        return df

    @bhom_analytics()
    def to_csv(self, csv_path: Path) -> Path:
        """Save this object as a csv file.

        Args:
            csv_path (Path):
                The path containing the CSV file.

        Returns:
            Path:
                The resultant CSV file.
        """
        csv_path = Path(csv_path)
        self.df.to_csv(csv_path)
        return csv_path

    @bhom_analytics()
    def wind_matrix(self) -> pd.DataFrame:
        """Calculate average wind speed and direction for each month and hour of day in a pandas DataFrame.
        Returns:
            pd.DataFrame:
                A DataFrame containing average wind speed and direction for each month and hour of day.
        """

        wind_directions = (
            (
                (
                    self.wd.groupby(
                        [self.wd.index.month, self.wd.index.hour], axis=0
                    ).apply(wind_direction_average)
                    # + 90
                )
                % 360
            )
            .unstack()
            .T
        )
        wind_directions.columns = [
            calendar.month_abbr[i] for i in wind_directions.columns
        ]
        wind_speeds = (
            self.ws.groupby([self.ws.index.month, self.ws.index.hour], axis=0)
            .mean()
            .unstack()
            .T
        )
        wind_speeds.columns = [calendar.month_abbr[i] for i in wind_speeds.columns]

        df = pd.concat(
            [wind_directions, wind_speeds], axis=1, keys=["direction", "speed"]
        )

        return df

    @bhom_analytics()
    def summarise(self) -> list[str]:
        """Generate a textual sumarry of the current object."""

        aa = self.frequency_table().sum(axis=0).idxmax()
        prevailing_count = self.frequency_table()[aa].sum()

        start_time = self.datetimes[0]
        end_time = self.datetimes[-1]
        prevailing_angle = self.prevailing(as_angle=True)[0]
        prevailing_cardinal = self.prevailing(as_angle=False)[0]
        prevailing_mean = self.ws.loc[self.wd == prevailing_angle].mean()
        prevailing_max = self.ws.loc[self.wd == prevailing_angle].max()

        # pylint: disable=line-too-long
        return [
            f"Between {start_time} and {end_time}, the prevailing wind direction is {prevailing_angle}° (or {prevailing_cardinal}) accounting for {prevailing_count} of {len(self.wd)} timesteps.",
            f"Prevailing wind average speed is {prevailing_mean:.2f}m/s, with a maximum of {prevailing_max:.2f}m/s.",
            f"Peak wind speeds were observed at {self.ws.idxmax()}, reaching {self.ws.max():.2f}m/s from {self.wd.loc[self.ws.idxmax()]}°.",
            f"{self.calm():.2%} of the time, wind speeds were calm (≤ 0.1m/s).",
        ]
        # pylint: enable=line-too-long

    ##################################
    # PLOTTING/VISUALISATION METHODS #
    ##################################

    @bhom_analytics()
    def plot_timeseries(self, ax: plt.Axes = None, color: str = "grey", **kwargs) -> plt.Axes:  # type: ignore
        """Create a simple line plot of wind speed.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. If None, the current axes will be used.
            color (str, optional):
                The color of the line to plot. Default is "blue".
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

        timeseries(self.ws, ax=ax, color=color)
        _, yhigh = ax.get_ylim()
        ax.set_ylim(0, yhigh)
        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))
        ax.set_ylabel(self.ws.name)

        return ax

    @bhom_analytics()
    def plot_windrose(
        self,
        ax: plt.Axes = None,
        data: list[float] = None,
        direction_bins: DirectionBins = DirectionBins(),
        data_bins: int | list[float] = 11,
        include_legend: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot a windrose for a collection of wind speeds and directions.

        Args:
            ax (plt.Axes, optional):
                The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
            data (list[float]):
                A collection of direction-associated data.
            direction_bins (DirectionBins, optional):
                A DirectionBins object.
            data_bins (Union[int, list[float]], optional):
                Bins to sort data into. Defaults to 11 bins between the min/max data values.
            include_legend (bool, optional):
                Set to True to include the legend. Defaults to True.
            **kwargs:
                Additional keyword arguments to pass to the function. These include:
                data_unit (str, optional):
                    The unit of the data to add to the legend. Defaults to None.
                ylim (tuple[float], optional):
                    The minimum and maximum values for the y-axis. Defaults to None.
                cmap (str, optional):
                    The name of the colormap to use. Defaults to "viridis".
                opening (float, optional):
                    The opening angle of the windrose. Defaults to 0.
                alpha (float, optional):
                    The alpha value of the windrose. Defaults to 1.
                title (str, optional):
                    A title for the plot. Defaults to None.

        Returns:
            plt.Axes:
                A Axes object.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        # remove 0-speed values
        local_w = self.filter_by_speed(min_speed=0.001)

        if not data:
            data = local_w.ws

        if len(data) != len(local_w.wd):
            raise ValueError(
                f"Length of the data ({len(data)}) must match the length of the wind-directions ({len(local_w.wd)})."
            )

        # HACK start - a fix introduced here to ensure that bar ends are curved when using a polar plot.
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        hist_ax = plt.Axes(fig, rect)
        hist_ax.bar(np.array([1]), np.array([1]))
        # HACK end

        opening = kwargs.pop("opening", 0.0)
        if opening >= 1 or opening < 0:
            raise ValueError("The opening must be between 0 and 1.")
        opening = 1 - opening

        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))

        # set data binning defaults
        _data_bins: list[float] = data_bins
        if isinstance(data_bins, int):
            _data_bins = np.linspace(min(data), max(data), data_bins + 1)

        # get colormap
        cmap = plt.get_cmap(
            kwargs.pop(
                "cmap",
                BEAUFORT_CATEGORIES.cmap,
            )
        )

        # bin input data
        thetas = np.deg2rad(direction_bins.midpoints)
        width = np.deg2rad(direction_bins.bin_width)
        binned_data = direction_bins.bin_data(local_w.wd, data)
        radiis = []
        for _, values in binned_data.items():
            radiis.append(np.histogram(a=values, bins=_data_bins)[0])
        _ = np.vstack(
            [[0] * len(direction_bins.midpoints), np.array(radiis).cumsum(axis=1).T]
        )[:-1].T
        colors = [cmap(i) for i in np.linspace(0, 1, len(_data_bins) - 1)]

        patches = []
        arr = []
        width = 2 * np.pi / len(thetas)
        for n, (_, _) in enumerate(binned_data.items()):
            _x = thetas[n] - (np.deg2rad(direction_bins.bin_width) / 2 * opening)
            _y = 0
            for m, radii in enumerate(radiis[n]):
                _color = colors[m]
                arr.extend(np.linspace(0, 1, len(_data_bins) - 1))
                patches.append(
                    Rectangle(
                        xy=(_x, _y),
                        width=width * opening,
                        height=radii,
                        alpha=kwargs.get("alpha", 1),
                    )
                )
                _y += radii
        pc = PatchCollection(patches, cmap=cmap, norm=plt.Normalize(0, 1))
        pc.set_array(arr)
        ax.add_collection(pc)

        format_polar_plot(ax)

        ax.set_ylim(
            kwargs.pop(
                "ylim", ax.set_ylim(0, np.ceil(max(sum(i) for i in radiis) / 10) * 10)
            )
        )

        # construct legend
        if include_legend:
            handles = [
                mpatches.Patch(color=colors[n], label=f"{i} to {j}")
                for n, (i, j) in enumerate(rolling_window(_data_bins, 2))
            ]
            _ = ax.legend(
                handles=handles,
                bbox_to_anchor=(1.1, 0.5),
                loc="center left",
                ncol=1,
                borderaxespad=0,
                frameon=False,
                fontsize="small",
                title=kwargs.pop("data_unit", None),
            )

        return ax

    @bhom_analytics()
    def plot_windhist(
        self,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot a 2D-histogram for a collection of wind speeds and directions.

        Args:
            ax (plt.Axes, optional):
                The axis to plot results on. Defaults to None.
            **kwargs:
                Additional keyword arguments to pass to the function. These include:
                ...

        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        if ax is None:
            ax = plt.gca()

        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))

        direction_bins = kwargs.pop("direction_bins", DirectionBins())
        speed_bins = np.linspace(0, self.ws.max() + 1, 21)
        mtx = self.frequency_table(
            direction_bins=direction_bins, speed_bins=speed_bins, density=True
        )

        # get edges for x's and y's
        x_width = direction_bins.bin_width
        x = np.sort(
            np.stack(
                [
                    direction_bins.midpoints - (x_width / 2),
                    direction_bins.midpoints + (x_width / 2),
                ]
            ).flatten()
        )
        y = np.array([np.array(i) for i in mtx.index]).flatten()
        z = mtx.values.repeat(2, axis=1).repeat(2, axis=0) * 100

        ax.set_xlabel(self.wd.name)
        ax.set_ylabel(self.ws.name)

        pcm = ax.pcolormesh(
            x,
            y,
            z[:-1, :-1],
            **kwargs,
        )
        cb = plt.colorbar(pcm, label="Frequency (%)")
        cb.outline.set_visible(False)

        return ax

    @bhom_analytics()
    def plot_windhist_radial(
        self,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot a wind histogram for a collection of wind speeds and directions.

        Args:
            ax (plt.Axes, optional):
                The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
            **kwargs:
                Additional keyword arguments to pass to the function. These include:
                cmap (str, optional):
                    The name of the colormap to use. Defaults to "viridis".
                direction_bins (DirectionBins, optional):
                    A DirectionBins object. Defaults to DirectionBins().
                speed_bins (list[float], optional):
                    A list of bins edges, between which wind speeds will be binned.
                title (str, optional):
                    Add a title. Defaults to None.

        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))

        # HACK start - a fix introduced here to ensure that bar ends are curved when using a polar plot.
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        hist_ax = plt.Axes(fig, rect)
        hist_ax.bar(np.array([1]), np.array([1]))
        # HACK end

        direction_bins = kwargs.pop("direction_bins", DirectionBins())
        speed_bins = kwargs.pop(
            "speed_bins",
            np.linspace(0, self.ws.max() + 1, 21),
        )
        cmap = plt.get_cmap(
            kwargs.pop(
                "cmap",
                "viridis",
            )
        )
        # colors = [cmap(i) for i in np.linspace(0, 1, len(speed_bins) - 1)]
        xx = self.frequency_table(
            direction_bins=direction_bins, speed_bins=speed_bins, density=True
        )

        thetas = np.deg2rad(direction_bins.midpoints)
        width = 2 * np.pi / len(thetas)
        alpha = kwargs.pop("alpha", 1)
        patches = []
        arr = []
        for n, (_, dir_values) in enumerate(xx.items()):
            _x = thetas[n] - np.deg2rad(direction_bins.bin_width) / 2
            _y = 0
            for speed_range, frequency_value in dir_values.items():
                height = speed_range[1] - speed_range[0]
                patches.append(
                    Rectangle(
                        xy=(_x, _y),
                        width=width,
                        height=height,
                        alpha=alpha,
                    )
                )
                arr.append(frequency_value)
                _y += height
        pc = PatchCollection(patches, cmap=cmap)
        pc.set_array(arr)
        ax.add_collection(pc)

        ax.set_ylim(0, xx.index[-1][-1])

        format_polar_plot(ax)

        return ax

    @bhom_analytics()
    def plot_wind_matrix(
        self,
        ax: plt.Axes = None,
        show_values: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Create a plot showing the annual wind speed and direction bins
        using the month_time_average method.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. If None, the current axes will be used.
            show_values (bool, optional):
                Whether to show values in the cells. Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to the pcolor function.

        Returns:
            plt.Axes:
                A matplotlib Axes object.

        """

        if ax is None:
            ax = plt.gca()

        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))

        df = self.wind_matrix()
        _wind_speeds = df["speed"]
        _wind_directions = df["direction"]

        if any(
            [
                _wind_speeds.shape != (24, 12),
                _wind_directions.shape != (24, 12),
                _wind_directions.shape != _wind_speeds.shape,
                not np.array_equal(_wind_directions.index, _wind_speeds.index),
                not np.array_equal(_wind_directions.columns, _wind_speeds.columns),
            ]
        ):
            raise ValueError(
                "The wind_speeds and wind_directions must cover all months of the "
                "year, and all hours of the day, and align with each other."
            )

        cmap = kwargs.pop("cmap", "Spectral_r")
        vmin = kwargs.pop("vmin", _wind_speeds.values.min())
        vmax = kwargs.pop("vmax", _wind_speeds.values.max())
        cbar_title = kwargs.pop("cbar_title", "m/s")
        norm = kwargs.pop("norm", Normalize(vmin=vmin, vmax=vmax, clip=True))
        mapper = kwargs.pop("mapper", ScalarMappable(norm=norm, cmap=cmap))

        pc = ax.pcolor(_wind_speeds, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        _x = -np.sin(np.deg2rad(_wind_directions.values))
        _y = -np.cos(np.deg2rad(_wind_directions.values))
        direction_matrix = angle_from_north([_x, _y])
        ax.quiver(
            np.arange(1, 13, 1) - 0.5,
            np.arange(0, 24, 1) + 0.5,
            _x * _wind_speeds.values / 2,
            _y * _wind_speeds.values / 2,
            pivot="mid",
            fc="white",
            ec="black",
            lw=0.5,
            alpha=0.5,
        )

        if show_values:
            for _xx, col in enumerate(_wind_directions.values.T):
                for _yy, _ in enumerate(col.T):
                    local_value = _wind_speeds.values[_yy, _xx]
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
                    # speed text
                    ax.text(
                        _xx + 1,
                        _yy + 1,
                        f"{_wind_speeds.values[_yy][_xx]:0.1f}m/s",
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

    @bhom_analytics()
    def plot_speed_frequency(
        self,
        ax: plt.Axes = None,
        percentiles: tuple[float] = (0.5, 0.95),
        **kwargs,
    ) -> plt.Axes:
        """Create a histogram showing wind speed frequency.

        Args:
            ax (plt.Axes, optional):
                The axes to plot this chart on. Defaults to None.

            percentiles (tuple[float], optional):
                The percentiles to plot. Defaults to (0.5, 0.95).
            **kwargs:
                Additional keyword arguments to pass to the self.frequency_table function.
                title (str, optional):
                    An optional title to give the chart. Defaults to None.
        Returns:
            plt.Axes: The axes object.
        """
        if ax is None:
            ax = plt.gca()

        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))

        kwargs.pop("include_counts", None)  # remove include_counts if present
        data = self.frequency_table(**kwargs).sum(axis=1)
        x_values = [np.mean(i) for i in data.index]
        y_values = data.values

        for percentile in percentiles:
            x = np.quantile(self.ws.values, percentile)
            ax.axvline(x, 0, 1, ls="--", lw=1, c="black", alpha=0.5)
            ax.text(x, 0, f"{percentile:0.0%}\n{x:0.2f}m/s", ha="left", va="bottom")

        ax.plot(x_values, y_values)

        ax.set_xlim(0, max(x_values))
        ax.set_ylim(0, max(y_values))

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Frequency")
        if kwargs.get("density", False):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.25)

        return ax

    @bhom_analytics()
    def plot_cumulative_density(
        self,
        ax: plt.Axes = None,
        percentiles: tuple[float] = (0.5, 0.95),
        **kwargs,
    ) -> plt.Axes:
        """Create a wind speed frequency cumulative density plot.

        Args:
            ax (plt.Axes, optional):
                The axes to plot this chart on. Defaults to None.
            percentiles (tuple[float], optional):
                The percentiles to plot. Defaults to (0.5, 0.95).
            **kwargs:
                Additional keyword arguments to pass to the self.cumulative_density_function function.
                title (str, optional):
                    An optional title to give the chart. Defaults to None.

        Returns:
            plt.Axes: The axes object.
        """

        if ax is None:
            ax = plt.gca()

        title = [kwargs.pop("title", None), repr(self)]
        ax.set_title("\n".join([i for i in title if i is not None]))

        data = self.cumulative_density_function(**kwargs).sum(axis=1)
        x_values = [np.mean(i) for i in data.index]
        y_values = data.values

        for percentile in percentiles:
            x = np.quantile(self.ws.values, percentile)
            ax.axvline(x, 0, 1, ls="--", lw=1, c="black", alpha=0.5)
            ax.text(x, 0, f"{percentile:0.0%}\n{x:0.2f}m/s", ha="left", va="bottom")

        ax.plot(x_values, y_values)

        ax.set_xlim(0, max(x_values))
        ax.set_ylim(0, max(y_values))

        ax.set_xlabel(self.ws.name)
        ax.set_ylabel("Frequency")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.25)

        return ax


# TODO - add Climate Consultant-style "wind wheel" plot here
# (http://2.bp.blogspot.com/-F27rpZL4VSs/VngYxXsYaTI/AAAAAAAACAc/yoGXmk13uf8/s1600/CC-graphics%2B-%2BWind%2BWheel.jpg)