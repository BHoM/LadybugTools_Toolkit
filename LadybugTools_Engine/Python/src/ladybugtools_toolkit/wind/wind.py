from __future__ import annotations

import calendar
import concurrent.futures
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.dt import DateTime
from ladybug.epw import EPW
from matplotlib.colors import Colormap
from tqdm import tqdm

from ..bhomutil.analytics import CONSOLE_LOGGER
from ..helpers import (
    OpenMeteoVariable,
    circular_weighted_mean,
    rolling_window,
    scrape_openmeteo,
    weibull_directional,
    weibull_pdf,
    wind_direction_average,
    wind_speed_at_height,
)
from ..ladybug_extension.analysis_period import (
    AnalysisPeriod,
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from ..plot import (
    radial_histogram,
    wind_cumulative_probability,
    wind_matrix,
    wind_speed_frequency,
    wind_timeseries,
    wind_windrose,
)
from .direction_bins import DirectionBins


class Wind:
    """An object containing historic, time-indexed wind data."""

    def __init__(
        self,
        wind_speeds: List[Union[int, float, np.number]],
        wind_directions: List[Union[int, float, np.number]],
        datetimes: Union[
            pd.DatetimeIndex, List[Union[datetime, np.datetime64, pd.Timestamp]]
        ],
        height_above_ground: float = 10,
    ):
        """
        Args:
            wind_speeds (List[Union[int, float, np.number]]):
                An iterable of wind speeds in m/s.
            wind_directions (List[Union[int, float, np.number]]):
                An iterable of wind directions in degrees from North (with North at 0-degrees).
            datetimes (Union[pd.DatetimeIndex, List[Union[datetime, np.datetime64, pd.Timestamp]]]):
                An iterable of datetime-like objects.
            height_above_ground (float, optional):
                The height above ground (in m) where the input wind speeds and directions were collected. Defaults to 10m.
        """
        self.validation(wind_speeds, wind_directions, datetimes, height_above_ground)

        self.datetimes: List[datetime] = pd.to_datetime(datetimes)
        self.wind_speeds = pd.Series(
            wind_speeds, index=self.datetimes, name="speed"
        ).sort_index()
        self.wind_directions = pd.Series(
            wind_directions, index=self.datetimes, name="direction"
        ).sort_index()
        self.df = pd.concat([self.ws, self.wd], axis=1)
        self.height_above_ground = height_above_ground

    ##################
    # DUNDER METHODS #
    ##################

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        """The printable representation of the given object"""
        return f"{self.__class__.__name__}({min(self.df.index):%Y-%m-%d} to {max(self.df.index):%Y-%m-%d}, n={len(self)} @{self.freq}, @{self.height_above_ground}m)"

    ##############
    # PROPERTIES #
    ##############

    @property
    def freq(self) -> str:
        """Return the inferred frequency of the datetimes associated with this object."""
        freq = pd.infer_freq(self.df.index)
        if freq is None:
            return "inconsistent"
        return freq

    @property
    def ws(self) -> pd.Series:
        """Convenience accessor for wind speeds as a time-indexed pd.Series object."""
        return self.wind_speeds

    @property
    def wd(self) -> pd.Series:
        """Convenience accessor for wind directions as a time-indexed pd.Series object."""
        return self.wind_directions

    @property
    def calm_datetimes(self) -> List[datetime]:
        """Return the datetimes where wind speed is < 0.1.

        Returns:
            List[datetime]:
                "Calm" wind datetimes.
        """
        return self.wind_speeds[self.wind_speeds <= 0.1].index.tolist()

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
    ) -> Wind:
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

        return cls(
            df[wind_speed_column].tolist(),
            df[wind_direction_column].tolist(),
            df.index.tolist(),
            height_above_ground,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        wind_speed_column: str,
        wind_direction_column: str,
        height_above_ground: float = 10,
    ) -> Wind:
        """Create a Wind object from a csv containing wind speed and direction columns.

        Args:
            csv_path (Union[str, Path]):
                The path to the CSV file containing speed and direction columns, and a datetime index.
            wind_speed_column (str):
                The name of the column where wind-speed data exists.
            wind_direction_column (str):
                The name of the column where wind-direction data exists.
            height_above_ground (float, optional):
                Defaults to 10m.
        """
        df = pd.read_csv(csv_path, index_col=0, header=0, parse_dates=True)
        return Wind.from_dataframe(
            df,
            wind_speed_column=wind_speed_column,
            wind_direction_column=wind_direction_column,
            height_above_ground=height_above_ground,
        )

    @classmethod
    def from_epw(cls, epw: Union[str, Path, EPW]) -> Wind:
        """Create a Wind object from an EPW file or object.

        Args:
            epw_file (Union[str, Path, EPW]):
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
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
    ) -> Wind:
        """Create a Wind object from data obtained from the Open-Meteo database of historic weather station data.

        Args:
            latitude (float):
                The latitude of the target site, in degrees.
            longitude (float):
                The longitude of the target site, in degrees.
            start_date (Union[datetime, str]):
                The start-date from which records will be obtained.
            end_date (Union[datetime, str]):
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
        wind_speeds = np.multiply(df["windspeed_10m (km/h)"], 0.277778).tolist()
        wind_directions = df["winddirection_10m (°)"].tolist()
        if len(wind_speeds) == 0 or len(wind_directions) == 0:
            raise ValueError(
                "OpenMeteo did not return any data for the given latitude, longitude and start/end dates."
            )
        datetimes = df.index.tolist()
        return Wind(wind_speeds, wind_directions, datetimes, 10)

    @classmethod
    def from_average(
        cls, wind_objects: List[Wind], weights: List[float] = None
    ) -> Wind:
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""

        # align collections so that intersection only is created
        df = pd.concat([i.df for i in wind_objects], axis=1)

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
        wd_avg = [circular_weighted_mean(i, weights) for r, i in df_wd.iterrows()]
        ws_avg = np.average(df_ws, axis=1, weights=weights)
        dts = df_ws.index

        # return the new averaged object
        return Wind(
            ws_avg,
            wd_avg,
            dts,
            np.average([i.height_above_ground for i in wind_objects], weights=weights),
        )

    ##################
    # STATIC METHODS #
    ##################

    @staticmethod
    def validation(
        wind_speeds: Any, wind_directions: Any, datetimes: Any, height_above_ground: Any
    ) -> None:
        """Ensure that values given conform to expected standards in order to create a valid Wind object.

        Args:
            wind_speeds (Any):
                Input for wind_speeds.
            wind_directions (Any):
                Input for wind_directions.
            datetimes (Any):
                Input for datetimes.
            height_above_ground (Any):
                Input for height_above_ground.
        """

        for arg_name, arg_value in locals().items():
            if arg_name == "height_above_ground":
                if arg_value <= 0:
                    raise ValueError(f"{arg_name} must be greater than 0")
            continue

            # check that arg is iterable, and not a string
            try:
                iter(arg_value)
            except TypeError as exc:
                raise ValueError(
                    f"{arg_name} must be single-dimension iterable"
                ) from exc

            if isinstance(arg_value, str):
                raise ValueError(f"{arg_name} must be single-dimension iterable")

            # check that iterable is flat
            if any(isinstance(el, (list, np.ndarray, tuple)) for el in arg_value):
                raise ValueError(f"{arg_name} must be single-dimension iterable")

            if arg_name in ["wind_speeds", "wind_directions"]:
                # check that inputs are numeric
                if not all(isinstance(el, (int, float, np.number)) for el in arg_value):
                    raise ValueError(f"{arg_name} must contain numeric values")

                if arg_name == "wind_speeds":
                    if min(arg_value) < 0:
                        raise ValueError(f"{arg_name} must be in m/s and ≥ 0")
                if arg_name == "wind_directions":
                    if (min(arg_value) < 0) or (max(arg_value) > 360):
                        raise ValueError(
                            f"{arg_name} must be in degrees (with north at 0) and in the range 0-360"
                        )

                # check that iterable contains no NaN values
                if sum(np.isnan(arg_value)) > 0:
                    raise ValueError(f"{arg_name} contains null values")

            else:
                # check that inputs are datetimes
                if not all(
                    isinstance(el, (datetime, pd.Timestamp, np.datetime64))
                    for el in arg_value
                ):
                    raise ValueError(f"{arg_name} must contain datetime-like values")

                # check that all datetimes are unique
                if len(datetimes) != len(np.unique(datetimes)):
                    raise ValueError(f"duplicate values are present in {arg_name}")

        # check height above ground validity
        if height_above_ground < 0.1:
            raise ValueError("height_above_ground must be >= 0.1")

        # check that inputs are the same shape/size
        if not len(wind_speeds) == len(wind_directions) == len(datetimes):
            raise ValueError(
                f"wind_speeds, wind_directions and datetimes must be the same length. ({len(wind_speeds)} != {len(wind_directions)} != {len(datetimes)})."
            )

    ###################
    # GENERAL METHODS #
    ###################

    def mean(self) -> float:
        """Return the mean wind speed for this object."""
        return self.ws.mean()

    def min(self) -> float:
        """Return the min wind speed for this object."""
        return self.ws.min()

    def max(self) -> float:
        """Return the max wind speed for this object."""
        return self.ws.max()

    def median(self) -> float:
        """Return the median wind speed for this object."""
        return self.ws.median()

    def calm(self, threshold: float = 0.1) -> float:
        """Return the proportion of timesteps "calm" (i.e. wind-speed ≤ 0.1)."""
        return (self.ws <= threshold).sum() / len(self.ws)

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

    def resample(self, rule: Union[pd.DateOffset, pd.Timedelta, str]) -> Wind:
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
            resampled_speeds.tolist(),
            resampled_directions.tolist(),
            resampled_datetimes,
            self.height_above_ground,
        )

    def prevailing(
        self,
        direction_bins: DirectionBins = DirectionBins(),
        n: int = 1,
        as_angle: bool = False,
    ) -> Union[List[float], List[str]]:
        """Calculate the prevailing wind direction/s for this object.

        Args:
            direction_bins (DirectionBins, optional):
                The number of wind directions to bin values into.
            n (int, optional):
                The number of prevailing directions to return. Default is 1.
            as_angle (bool, optional):
                Return directions as an angle or as a cardinal direction. Default is False to return cardinal direction.

        Returns:
            Union[List[float], List[str]]:
                A list of wind directions.
        """

        return direction_bins.prevailing(self.wd.tolist(), n, as_angle)

    def probabilities(
        self,
        direction_bins: DirectionBins = DirectionBins(),
        percentiles: Tuple[float, float] = (0.5, 0.95),
    ) -> pd.DataFrame:
        """Calculate the probabilities of wind speeds at the given percentiles, for the direction bins specified.

        Args:
            direction_bins (DirectionBins, optional):
                A DirectionBins object. Defaults to DirectionBins().
            percentiles (Tuple[float, float], optional):
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

    def filter_by_analysis_period(
        self,
        analysis_period: Union[AnalysisPeriod, Tuple[AnalysisPeriod]],
    ) -> Wind:
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

        return Wind.from_dataframe(
            df,
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=self.height_above_ground,
        )

    def filter_by_boolean_mask(self, mask: Tuple[bool]) -> Wind:
        """Filter the current object by a boolean mask.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if len(mask) != len(self.ws):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )

        return Wind(
            self.ws[mask].tolist(),
            self.wd[mask].tolist(),
            self.datetimes[mask],
            self.height_above_ground,
        )

    def filter_by_time(
        self,
        months: Tuple[float] = tuple(range(1, 13, 1)),  # type: ignore
        hours: Tuple[int] = tuple(range(0, 24, 1)),  # type: ignore
        years: Tuple[int] = tuple(range(1900, 2100, 1)),
    ) -> Wind:
        """Filter the current object by month and hour.

        Args:
            months (List[int], optional):
                A list of months. Defaults to all months.
            hours (List[int], optional):
                A list of hours. Defaults to all hours.
            years (Tuple[int], optional):
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
        return Wind(
            self.ws.iloc[indices].tolist(),
            self.wd.iloc[indices].tolist(),
            self.datetimes[indices],
            self.height_above_ground,
        )

    def filter_by_direction(
        self, left_angle: float = 0, right_angle: float = 360, inclusive: bool = True
    ) -> Wind:
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

        return Wind(
            self.ws[mask].tolist(),
            self.wd[mask].tolist(),
            self.datetimes[mask],
            self.height_above_ground,
        )

    def filter_by_speed(
        self, min_speed: float = 0, max_speed: float = 999, inclusive: bool = True
    ) -> Wind:
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

        return Wind(
            self.ws[mask].tolist(),
            self.wd[mask].tolist(),
            self.datetimes[mask],
            self.height_above_ground,
        )

    def frequency_table(
        self,
        speed_bins: Tuple[float] = (0, 0.1) + tuple(range(2, 39, 2)) + (np.inf,),
        direction_bins: DirectionBins = DirectionBins(),
        density: bool = False,
        include_counts: bool = False,
    ) -> pd.DataFrame:
        """Create a table with wind direction per rows, and wind speed per column.

        Args:
            speed_bins (List[float], optional):
                A list of bins edges, between which wind speeds will be binned. Defaults to (0, 0.1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 99.9).
            directions (int, optional):
                The number of directions into which wind directions should be binned. Defaults to 8, centered around 0/north.
            density (bool, optional):
                Set to True to return density (0-1) instead of count. Defaults to False.
            include_counts (bool, optional):
                Include counts for each row and columns if True. Defaults to False.

        Returns:
            pd.DataFrame:
                A frequency table.
        """

        if direction_bins.directions <= 3:
            raise ValueError(
                "At least 4 directions should be specified in the DirectionBins object."
            )

        df = self.df

        # bin data using speed bins and direction
        ds = []
        speed_bin_labels = []
        for low, high in rolling_window(speed_bins, 2):
            speed_bin_labels.append((low, high))
            mask = (df.speed >= low) & (df.speed < high)
            binned = direction_bins.bin_data(df.direction[mask])
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

    def weibull_pdf(self) -> Tuple[float]:
        """Calculate the parameters of an exponentiated Weibull continuous random variable.

        Returns:
            x (float):
                Fixed shape parameter (1).
            k (float):
                Shape parameter 1.
            λ (float):
                Scale parameter.
            α (float):
                Shape parameter 2.
        """
        return weibull_pdf(self.ws.tolist())

    def weibull_directional(
        self, direction_bins: DirectionBins = DirectionBins()
    ) -> pd.DataFrame:
        """Calculate directional weibull coefficients for the given number of directions.

        Args:
            directions (int, optional):
                The number of directions into which wind directions should be binned. Defaults to 8.

        Returns:
            pd.DataFrame:
                A DataFrame object.
        """
        binned_data = direction_bins.bin_data(self.wd.tolist(), self.ws.tolist())
        return weibull_directional(binned_data)

    def to_height(
        self,
        target_height: float,
        terrain_roughness_length: float = 1,
        log_function: bool = True,
    ) -> Wind:
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
            reference_value=self.ws,  # type: ignore
            reference_height=self.height_above_ground,
            target_height=target_height,
            terrain_roughness_length=terrain_roughness_length,
            log_function=log_function,
        )
        return Wind(ws.tolist(), self.wd.tolist(), self.datetimes, target_height)  # type: ignore

    def apply_directional_factors(
        self, direction_bins: DirectionBins, factors: Tuple[float]
    ) -> Wind:
        """Adjust wind speed values by a set of factors per direction.

        Args:
            directions (int, optional):
                The number of directions to bin wind-directions into. Defaults to 8.
            factors (Tuple[float], optional):
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
            ws.tolist(), self.wd.tolist(), self.datetimes, self.height_above_ground
        )

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

    def to_csv(self, csv_path: Union[str, Path]) -> Path:
        """Save this object as a csv file.

        Args:
            csv_path (Union[str, Path]):
                The path containing the CSV file.

        Returns:
            Path:
                The resultant CSV file.
        """
        csv_path = Path(csv_path)
        self.df.to_csv(csv_path)
        return csv_path

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

    ##################################
    # PLOTTING/VISUALISATION METHODS #
    ##################################

    def plot_windrose(
        self,
        direction_bins: DirectionBins = DirectionBins(),
        bins: List[float] = None,
        include_legend: bool = True,
        include_percentages: bool = False,
        title: str = None,
        cmap: Union[Colormap, str] = "YlGnBu",
        calm_threshold: float = 0.1,
        include_calm_threshold: bool = True,
    ) -> plt.Figure:  # type: ignore
        """Create a windrose.

        Args:
            direction_bins (DirectionBins, optional):
                A DirectionBins object.
            bins (List[float], optional):
                Bins to sort data into.
            include_legend (bool, optional):
                Set to True to include the legend. Defaults to True.
            include_percentages (bool, optional):
                Add bin totals as % to rose. Defaults to False.
            title (str, optional):
                Add a custom title to this plot.
            cmap (Union[Colormap, str], optional):
                Use a custom colormap. Defaults to "YlGnBu".

        Returns:
            plt.Figure:
                A Figure object.
        """

        # remove calm hours and store % calm
        calm_percentage = self.calm(calm_threshold)
        new_w = self.filter_by_speed(
            min_speed=calm_threshold, max_speed=np.inf, inclusive=False
        )

        if title is not None:
            if include_calm_threshold:
                ti = f"{title}\n{calm_percentage:0.1%} calm (≤ {calm_threshold}m/s)"
            else:
                ti = f"{title}"
        else:
            if include_calm_threshold:
                ti = f"{self}\n{calm_percentage:0.1%} calm (≤ {calm_threshold}m/s)"
            else:
                ti = f"{self}"

        return wind_windrose(
            wind_direction=new_w.wd.tolist(),
            data=new_w.ws.tolist(),
            direction_bins=direction_bins,
            data_bins=bins,
            cmap=cmap,
            title=ti,
            include_legend=include_legend,
            include_percentages=include_percentages,
        )

    def plot_windroses_parallel(
        self,
        analysis_periods: List[AnalysisPeriod],
        save_directory: Path,
        prepend_file: str = "parallel",
        direction_bins: DirectionBins = DirectionBins(),
        bins: List[float] = None,
        include_legend: bool = True,
        include_percentages: bool = False,
        cmap: Union[Colormap, str] = "YlGnBu",
        calm_threshold: float = 0.1,
    ) -> None:
        """Generate a series of windroses in parallel for a set of analysis periods. This is useful for comparing windroses for different periods of time.

        Args:
            analysis_periods (List[AnalysisPeriod]):
                A list of AnalysisPeriod objects.
            save_directory (Path):
                The directory to save the windroses to.
            prepend_file (str, optional):
                An identifier to prepedn the resultant images with. Defaults to "parallel".
            direction_bins (DirectionBins, optional):
                A DirectionBins object.
            bins (List[float], optional):
                Bins to sort data into.
            include_legend (bool, optional):
                Set to True to include the legend. Defaults to True.
            include_percentages (bool, optional):
                Add bin totals as % to rose. Defaults to False.
            title (str, optional):
                Add a custom title to this plot.
            cmap (Union[Colormap, str], optional):
                Use a custom colormap. Defaults to "YlGnBu".

        """
        save_directory = Path(save_directory)
        if not save_directory.is_dir():
            raise ValueError(f"{save_directory} is not a directory.")
        if not save_directory.exists():
            raise ValueError(f"{save_directory} does not exist.")

        def _savefig(obj: Wind, ap: AnalysisPeriod):
            sp = (
                save_directory
                / f"{prepend_file}_{describe_analysis_period(ap, save_path=True)}.png"
            )
            f = obj.filter_by_analysis_period(ap).plot_windrose(
                direction_bins,
                bins,
                include_legend,
                include_percentages,
                describe_analysis_period(ap),
                cmap,
                calm_threshold,
            )
            f.savefig(
                sp,
                dpi=300,
                transparent=True,
            )
            plt.close(f)
            if not sp.exists():
                raise RuntimeError(f"Failed to save {sp}")
            return sp

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ap in analysis_periods:
                results.append(executor.submit(_savefig, self, ap))

        for result in results:
            print(result.result())

    def plot_windhist(
        self,
        direction_bins: DirectionBins = DirectionBins(),
        speed_bins: List[float] = None,
        density: bool = False,
        include_cbar: bool = True,
        title: str = None,
        cmap: Union[Colormap, str] = "magma_r",
        calm_threshold: float = 0.1,
    ) -> plt.Figure:
        """_summary_

        Args:
            direction_bins (DirectionBins, optional): _description_. Defaults to DirectionBins().
            speed_bins (List[float], optional): _description_. Defaults to None.
            density (bool, optional): _description_. Defaults to False.
            include_cbar (bool, optional): _description_. Defaults to True.
            title (str, optional): _description_. Defaults to None.
            cmap (Union[Colormap, str], optional): _description_. Defaults to "magma_r".

        Returns:
            plt.Figure: _description_
        """

        # remove calm hours and store % calm
        calm_percentage = self.calm(calm_threshold)
        new_w = self.filter_by_speed(
            min_speed=calm_threshold, max_speed=np.inf, inclusive=False
        )

        if speed_bins is None:
            _low = int(np.floor(new_w.min()))
            _high = int(np.ceil(new_w.max()))
            speed_bins = np.linspace(_low, _high, (_high - _low) + 1)

        if title is not None:
            ti = f"{title}\n{calm_percentage:0.1%} calm (≤ {calm_threshold}m/s)"
        else:
            ti = f"{self}\n{calm_percentage:0.1%} calm (≤ {calm_threshold}m/s)"

        frequency_table = new_w.frequency_table(
            speed_bins, direction_bins, density=density, include_counts=False
        )

        direction_angles = np.deg2rad(direction_bins.midpoints)
        radial_bins = [np.mean(i) for i in frequency_table.index]

        if density:
            cmap_label = "Frequency"  #
            cbar_freq = True
        else:
            cmap_label = "n-occurences"
            cbar_freq = False

        fig = radial_histogram(
            direction_angles,
            radial_bins,
            frequency_table.values,
            cmap=cmap,
            include_labels=False,
            include_cbar=include_cbar,
            cmap_label=cmap_label,
            cbar_freq=cbar_freq,
            title=ti,
        )

        return fig

    def plot_wind_matrix(self, title: str = None, cmap: Union[Colormap, str] = "YlGnBu", show_values: bool = False, speed_lims: Tuple[float] = None) -> plt.Figure:  # type: ignore
        """Create a plot showing the annual wind speed and direction bins using the month_time_average method."""
        df = self.wind_matrix()

        wind_speed_bins = df["speed"]
        wind_direction_bins = df["direction"]

        if title is None:
            title = f"{self}"
        else:
            title = f"{self}\n{title}"

        return wind_matrix(
            wind_speeds=wind_speed_bins,
            wind_directions=wind_direction_bins,
            cmap=cmap,
            title=title,
            show_values=show_values,
            speed_lims=speed_lims,
        )

    def plot_timeseries(self, color: str = "grey") -> plt.Figure:  # type: ignore
        """Create a simple line plot of wind speed.

        Args:
            color (str, optional):
                The color of the line to plot. Default is "blue".

        Returns:
            plt.Figure:
                A Figure object.

        """
        return wind_timeseries(self.ws, color=color, title=str(self))

    def plot_speed_frequency(self, title: str = None) -> plt.Figure:  # type: ignore
        """Create a histogram showing wind speed frequency"""

        if title is None:
            title = str(self)
        else:
            title = f"{self}\n{title}"

        speed_bins = np.linspace(min(self.ws), np.quantile(self.ws, 0.999), 16)
        percentiles = (0.5, 0.95)

        return wind_speed_frequency(
            self.ws.tolist(),
            speed_bins=speed_bins,
            weibull=self.weibull_pdf(),
            percentiles=percentiles,
            title=title,
        )

    def plot_cumulative_probability(
        self, percentiles: Tuple[float] = (0.5, 0.95), title: str = None
    ) -> plt.Figure:  # type: ignore
        """Create a cumulative probability plot"""

        if title is None:
            title = str(self)
        else:
            title = f"{self}\n{title}"

        return wind_cumulative_probability(
            self.ws.tolist(),
            speed_bins=np.linspace(0, 25, 50).tolist(),
            percentiles=percentiles,
            title=title,
        )

    # def plot_windrose_matrix(
    #     self,
    #     month_bins: Tuple[List[int]],
    #     hour_bins: Tuple[List[int]],
    #     direction_bins: DirectionBins = DirectionBins(),
    #     data_bins: List[float] = None,
    #     title: str = None,
    # ) -> plt.Figure:
    #     """Create a plot showing the annual wind direction in a matrix of month and hour bins."""
    #     fig = windrose_matrix(
    #         wind_direction=self.wd,
    #         data=self.ws,
    #         month_bins=month_bins,
    #         hour_bins=hour_bins,
    #         data_bins=data_bins,
    #         direction_bins=direction_bins,
    #         cmap="YlGnBu",
    #         title=title if title is not None else str(self),
    #     )
    #     return fig
