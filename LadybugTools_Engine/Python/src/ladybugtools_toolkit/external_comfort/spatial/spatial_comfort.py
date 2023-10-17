"""Methods for handling SpatialMetric results for spatial comfort assessments."""
# pylint: disable=broad-exception-caught,W0212
# pylint: disable=E0401
import calendar
import contextlib
import copy
import io
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# pyling: enable=E0401

import numpy as np
import pandas as pd
from honeybee.model import Model
from ladybug.analysisperiod import AnalysisPeriod
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...helpers import sanitise_string
from ...honeybee_extension.results import load_ill, load_pts, load_res, make_annual
from ...ladybug_extension.analysisperiod import (
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from ...ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ...ladybug_extension.epw import seasonality_from_month, sun_position_list
from ...plot._utci import (
    utci_heatmap_difference,
    utci_heatmap_histogram,
)
from ...plot.utilities import create_triangulation
from ...plot._sunpath import sunpath
from .._simulatebase import SimulationResult
from ...bhom import CONSOLE_LOGGER

# from ..simulate import direct_sun_hours as dsh
from ..utci import utci, utci_parallel
from .calculate import (
    rwdi_london_thermal_comfort_category,
    shaded_unshaded_interpolation,
)
from .cfd import spatial_wind_speed
from .metric import SpatialMetric


@dataclass(init=True, repr=True, eq=True)
class SpatialComfort:
    """A SpatialComfort object, used to calculate spatial UTCI.

    Args:
        spatial_simulation_directory (Path): A directory path containing SkyView and Annual
            Irradiance simulation results.
        simulation_result (SimulationResult): A results object containing
            pre-simulated MRT values.

    Returns:
        SpatialComfort: A SpatialComfort object.
    """

    spatial_simulation_directory: Path = field(init=True, repr=True, compare=False)
    simulation_result: SimulationResult = field(init=True, repr=True, compare=False)

    _model: Model = field(init=False, compare=False, repr=False, default=None)
    _points: pd.DataFrame = field(init=False, compare=False, repr=False, default=None)
    _dry_bulb_temperature_epw: pd.DataFrame = field(
        init=False, compare=False, repr=False, default=None
    )
    _relative_humidity_epw: pd.DataFrame = field(
        init=False, compare=False, repr=False, default=None
    )
    _wind_speed_epw: pd.DataFrame = field(
        init=False, compare=False, repr=False, default=None
    )
    _wind_direction_epw: pd.DataFrame = field(
        init=False, compare=False, repr=False, default=None
    )
    _irradiance_total: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )
    _irradiance_direct: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )
    _irradiance_diffuse: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )
    _sky_view: pd.DataFrame = field(init=False, repr=False, compare=False, default=None)
    _mean_radiant_temperature_interpolated: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )
    _wind_speed_cfd: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )
    _universal_thermal_climate_index_interpolated: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )
    _universal_thermal_climate_index_calculated: pd.DataFrame = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self):
        self.spatial_simulation_directory = Path(self.spatial_simulation_directory)

        if not spatial_comfort_possible(self.spatial_simulation_directory):
            raise RuntimeError(
                "The spatial_simulation_directory given cannot be processed to create a SpatialComfort object."
            )

        # calculate baseline UTCI for shaded/unshaded
        self._unshaded_utci = utci(
            self.simulation_result.epw.dry_bulb_temperature,
            self.simulation_result.epw.relative_humidity,
            self.simulation_result.UnshadedMeanRadiantTemperature,
            self.simulation_result.epw.wind_speed,
        )
        self._shaded_utci = utci(
            self.simulation_result.epw.dry_bulb_temperature,
            self.simulation_result.epw.relative_humidity,
            self.simulation_result.ShadedMeanRadiantTemperature,
            self.simulation_result.epw.wind_speed,
        )

        # create directory for generated plots
        self._plot_directory = self.spatial_simulation_directory / "plots"
        self._plot_directory.mkdir(exist_ok=True, parents=True)

        # create default triangulation
        self._triangulation = create_triangulation(
            self.points.x.values,
            self.points.y.values,
            alpha=3,
            max_iterations=250,
            increment=0.01,
        )

        # add default analysis periods
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.analysis_periods: list[AnalysisPeriod] = []
            for month, day in [[3, 21], [6, 21], [9, 21], [12, 21]]:
                for start_hour, end_hour in [[8, 20], [0, 23]]:
                    for timestep in [1, 12]:
                        self.analysis_periods.append(
                            AnalysisPeriod(
                                st_month=month,
                                end_month=month,
                                st_day=day,
                                end_day=day,
                                st_hour=start_hour,
                                end_hour=end_hour,
                                timestep=timestep,
                            ),
                        )
            for start_month, end_month in [[12, 2], [3, 5], [6, 8], [9, 11]]:
                for start_hour, end_hour in [[8, 20], [0, 23]]:
                    for timestep in [1, 12]:
                        self.analysis_periods.append(
                            AnalysisPeriod(
                                st_month=start_month,
                                end_month=end_month,
                                st_hour=start_hour,
                                end_hour=end_hour,
                                timestep=timestep,
                            ),
                        )
            # custom time periods - Toronto
            for start_month, end_month in [[10, 10]]:
                for start_hour, end_hour in [[8, 20], [0, 23]]:
                    for timestep in [1, 12]:
                        self.analysis_periods.append(
                            AnalysisPeriod(
                                st_month=start_month,
                                end_month=end_month,
                                st_hour=start_hour,
                                end_hour=end_hour,
                                timestep=timestep,
                            ),
                        )

            # add hottest/coldest (avg) days
            for day in [self.hottest_day, self.coldest_day]:
                for start_hour, end_hour in [[8, 20], [0, 23]]:
                    for timestep in [1, 12]:
                        self.analysis_periods.append(
                            AnalysisPeriod(
                                st_month=day.month,
                                end_month=day.month,
                                st_day=day.day,
                                end_day=day.day,
                                st_hour=start_hour,
                                end_hour=end_hour,
                                timestep=timestep,
                            ),
                        )

        # wrap methods within this class
        # commented out here to avoid BHoMObject running all methods upon instantiation!
        # super().__post_init__()

    def __repr__(self) -> str:
        return f"{self.spatial_simulation_directory.name}"

    @property
    def model(self) -> Model:
        """Return the mode associated with this object."""
        if self._model is None:
            self._model = Model.from_hbjson(
                list(self.spatial_simulation_directory.glob("**/*.hbjson"))[0]
            )
        return self._model

    @property
    def points(self) -> pd.DataFrame:
        """Obtain the point locations for this object."""
        if self._points is None:
            metric = SpatialMetric.POINTS
            save_path = metric.filepath(self.spatial_simulation_directory)
            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                self._points = pd.read_parquet(save_path)
                return self._points

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            files = list(
                (
                    self.spatial_simulation_directory / "sky_view" / "model" / "grid"
                ).glob("*.pts")
            )
            df = load_pts(files).droplevel(0, axis=1)
            df.to_parquet(save_path)
            self._points = df
        return self._points

    @property
    def dry_bulb_temperature_epw(self) -> pd.DataFrame:
        """Obtain the DBT values from the weatherfile."""
        if self._dry_bulb_temperature_epw is None:
            metric = SpatialMetric.DBT_EPW
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._dry_bulb_temperature_epw = df
                return self._dry_bulb_temperature_epw

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            series = collection_to_series(
                self.simulation_result.epw.dry_bulb_temperature
            )
            df = pd.DataFrame(
                np.tile(series.values, (len(self.points.x.values), 1)).T,
                index=series.index,
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._dry_bulb_temperature_epw = df
        return self._dry_bulb_temperature_epw

    @property
    def relative_humidity_epw(self) -> pd.DataFrame:
        """Obtain the RH values from the weatherfile."""
        if self._relative_humidity_epw is None:
            metric = SpatialMetric.RH_EPW
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._relative_humidity_epw = df
                return self._relative_humidity_epw

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            series = collection_to_series(self.simulation_result.epw.relative_humidity)
            df = pd.DataFrame(
                np.tile(series.values, (len(self.points.x.values), 1)).T,
                index=series.index,
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._relative_humidity_epw = df
        return self._relative_humidity_epw

    @property
    def wind_speed_epw(self) -> pd.DataFrame:
        """Obtain the WS values from the weatherfile."""
        if self._wind_speed_epw is None:
            metric = SpatialMetric.WS_EPW
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._wind_speed_epw = df
                return self._wind_speed_epw

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            series = collection_to_series(self.simulation_result.epw.wind_speed)
            df = pd.DataFrame(
                np.tile(series.values, (len(self.points.x.values), 1)).T,
                index=series.index,
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._wind_speed_epw = df
        return self._wind_speed_epw

    @property
    def wind_direction_epw(self) -> pd.DataFrame:
        """Obtain the WD values from the weatherfile."""
        if self._wind_direction_epw is None:
            metric = SpatialMetric.WD_EPW
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._wind_direction_epw = df
                return self._wind_direction_epw

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            series = collection_to_series(self.simulation_result.epw.wind_direction)
            df = pd.DataFrame(
                np.tile(series.values, (len(self.points.x.values), 1)).T,
                index=series.index,
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._wind_direction_epw = df
        return self._wind_direction_epw

    @property
    def irradiance_total(self) -> pd.DataFrame:
        """Obtain the Total Irradiance values for this object."""
        if self._irradiance_total is None:
            metric = SpatialMetric.RAD_TOTAL
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._irradiance_total = df
                return self._irradiance_total

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            ill_files = list(
                (
                    self.spatial_simulation_directory
                    / "annual_irradiance"
                    / "results"
                    / "total"
                ).glob("*.ill")
            )
            df = (
                make_annual(load_ill(ill_files))
                .fillna(0)
                .clip(lower=0)
                .droplevel(0, axis=1)
                .round(0)
            )
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._irradiance_total = df
        return self._irradiance_total

    @property
    def irradiance_direct(self) -> pd.DataFrame:
        """Obtain the Direct Irradiance values for this object."""
        if self._irradiance_direct is None:
            metric = SpatialMetric.RAD_DIRECT
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._irradiance_direct = df
                return self._irradiance_direct

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            ill_files = list(
                (
                    self.spatial_simulation_directory
                    / "annual_irradiance"
                    / "results"
                    / "direct"
                ).glob("*.ill")
            )
            df = (
                make_annual(load_ill(ill_files))
                .fillna(0)
                .clip(lower=0)
                .droplevel(0, axis=1)
                .round(0)
            )
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._irradiance_direct = df
        return self._irradiance_direct

    @property
    def irradiance_diffuse(self) -> pd.DataFrame:
        """Obtain the Diffuse Irradiance values for this object."""
        if self._irradiance_diffuse is None:
            metric = SpatialMetric.RAD_DIFFUSE
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._irradiance_diffuse = df
                return self._irradiance_diffuse

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            df: pd.DataFrame = (
                (self.irradiance_total - self.irradiance_direct).clip(lower=0).round(0)
            )
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._irradiance_diffuse = df
        return self._irradiance_diffuse

    @property
    def sky_view(self) -> pd.DataFrame:
        """Obtain the Sky View values for this object."""
        if self._sky_view is None:
            metric = SpatialMetric.SKY_VIEW
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                self._sky_view = df
                return self._sky_view

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            try:
                files = list(
                    (
                        self.spatial_simulation_directory
                        / "sky_view"
                        / "results"
                        / "sky_view"
                    ).glob("*.res")
                )
                df = load_res(files).clip(lower=0, upper=100).round(2)
            except Exception as _:
                files = list(
                    (self.spatial_simulation_directory / "sky_view" / "results").glob(
                        "*.res"
                    )
                )
                df = load_res(files).clip(lower=0, upper=100).round(2)

            df.to_parquet(save_path)
            self._sky_view = df
        return self._sky_view

    @property
    def mean_radiant_temperature_interpolated(self) -> pd.DataFrame:
        """Obtain the MRT values (using a rapid interpolation method)."""
        if self._mean_radiant_temperature_interpolated is None:
            metric = SpatialMetric.MRT_INTERPOLATED
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._mean_radiant_temperature_interpolated = df
                return self._mean_radiant_temperature_interpolated

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            df = shaded_unshaded_interpolation(
                unshaded_value=self.simulation_result.UnshadedMeanRadiantTemperature.values,
                shaded_value=self.simulation_result.ShadedMeanRadiantTemperature.values,
                total_irradiance=self.irradiance_total.values,
                sky_view=self.sky_view.squeeze().values,
                sun_up=[
                    i.altitude > 0
                    for i in sun_position_list(self.simulation_result.epw)
                ],
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._mean_radiant_temperature_interpolated = df
        return self._mean_radiant_temperature_interpolated

    @property
    def wind_speed_cfd(self) -> pd.DataFrame:
        """Obtain the WS values from a CFD simulation."""
        if self._wind_speed_cfd is None:
            metric = SpatialMetric.WS_CFD
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._wind_speed_cfd = df
                return self._wind_speed_cfd

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            df = spatial_wind_speed(
                self.spatial_simulation_directory, self.simulation_result.epw
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._wind_speed_cfd = df
        return self._wind_speed_cfd

    @property
    def universal_thermal_climate_index_interpolated(self) -> pd.DataFrame:
        """Obtain UTCI values (using a rapid interpolation method) for this object."""
        if self._universal_thermal_climate_index_interpolated is None:
            metric = SpatialMetric.UTCI_INTERPOLATED
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._universal_thermal_climate_index_interpolated = df
                return self._universal_thermal_climate_index_interpolated

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            df = shaded_unshaded_interpolation(
                unshaded_value=self._unshaded_utci.values,
                shaded_value=self._shaded_utci.values,
                total_irradiance=self.irradiance_total.values,
                sky_view=self.sky_view.squeeze().values,
                sun_up=[
                    i.altitude > 0
                    for i in sun_position_list(self.simulation_result.epw)
                ],
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._universal_thermal_climate_index_interpolated = df
        return self._universal_thermal_climate_index_interpolated

    @property
    def universal_thermal_climate_index_calculated(self) -> pd.DataFrame:
        """Obtain UTCI values (using a point-wise calculation method)."""
        if self._universal_thermal_climate_index_calculated is None:
            metric = SpatialMetric.UTCI_CALCULATED
            save_path = metric.filepath(self.spatial_simulation_directory)

            if save_path.exists():
                CONSOLE_LOGGER.info(f"[{self}] - Loading {metric.description()}")
                df = pd.read_parquet(save_path)
                df.columns = df.columns.astype(int)
                self._universal_thermal_climate_index_calculated = df
                return self._universal_thermal_climate_index_calculated

            CONSOLE_LOGGER.info(f"[{self}] - Generating {metric.description()}")
            dbt = self.dry_bulb_temperature_epw
            rh = self.relative_humidity_epw

            try:
                ws = self.wind_speed_cfd
            except Exception as _:
                try:
                    ws = self.wind_speed_epw
                except Exception as exc_inner:
                    raise exc_inner

            df = pd.DataFrame(
                utci_parallel(
                    dbt.values,
                    self.mean_radiant_temperature_interpolated.values,
                    ws.values,
                    rh.values,
                ),
                index=self.mean_radiant_temperature_interpolated.index,
            ).round(2)
            df.columns = df.columns.astype(str)
            df.to_parquet(save_path)
            self._universal_thermal_climate_index_calculated = df
        return self._universal_thermal_climate_index_calculated

    @property
    def points_xy(self) -> np.ndarray:
        """Get the points associated with this object as an array of [[x, y], [x, y], ...]"""
        return np.stack([self.points.x.values, self.points.y.values], axis=1)

    @property
    def points_xyz(self) -> np.ndarray:
        """Get the points associated with this object as an array of [[x, y, z], [x, y, z], ...]"""
        return np.stack(
            [self.points.x.values, self.points.y.values, self.points.z.values], axis=1
        )

    @property
    def coldest_day(self) -> datetime:
        """The coldest day in the year associated with this case (average for all hours in day),
        within the winter period."""
        seasons = seasonality_from_month(self.simulation_result.epw)
        return (
            collection_to_series(self.simulation_result.epw.dry_bulb_temperature)
            .loc[seasons == "Winter"]
            .resample("D")
            .mean()
            .idxmin()
            .to_pydatetime()
        )

    @property
    def hottest_day(self) -> datetime:
        """The hottest day in the year associated with this case (average for all hours in day),
        within the summer period."""
        seasons = seasonality_from_month(self.simulation_result.epw)
        return (
            collection_to_series(self.simulation_result.epw.dry_bulb_temperature)
            .loc[seasons == "Summer"]
            .resample("D")
            .mean()
            .idxmax()
            .to_pydatetime()
        )

    def filter_by_point_indices(self, indices: list[int]) -> "SpatialComfort":
        """Filter this object by a list of indices."""

        new_obj = copy.copy(self)

        if new_obj._points is not None:
            new_obj._points = new_obj.points.iloc[indices]

        if new_obj._sky_view is not None:
            new_obj._sky_view = new_obj.sky_view.iloc[indices]

        if new_obj._dry_bulb_temperature_epw is not None:
            new_obj._dry_bulb_temperature_epw = new_obj.dry_bulb_temperature_epw.iloc[
                :, indices
            ]

        if new_obj._relative_humidity_epw is not None:
            new_obj._relative_humidity_epw = new_obj.relative_humidity_epw.iloc[
                :, indices
            ]

        if new_obj._wind_speed_epw is not None:
            new_obj._wind_speed_epw = new_obj.wind_speed_epw.iloc[:, indices]

        if new_obj._wind_direction_epw is not None:
            new_obj._wind_direction_epw = new_obj.wind_direction_epw.iloc[:, indices]

        if new_obj._irradiance_total is not None:
            new_obj._irradiance_total = new_obj.irradiance_total.iloc[:, indices]

        if new_obj._irradiance_direct is not None:
            new_obj._irradiance_direct = new_obj.irradiance_direct.iloc[:, indices]

        if new_obj._irradiance_diffuse is not None:
            new_obj._irradiance_diffuse = new_obj.irradiance_diffuse.iloc[:, indices]

        if new_obj._mean_radiant_temperature_interpolated is not None:
            new_obj._mean_radiant_temperature_interpolated = (
                new_obj.mean_radiant_temperature_interpolated.iloc[:, indices]
            )

        if new_obj._wind_speed_cfd is not None:
            new_obj._wind_speed_cfd = new_obj.wind_speed_cfd.iloc[:, indices]

        if new_obj._universal_thermal_climate_index_interpolated is not None:
            new_obj._universal_thermal_climate_index_interpolated = (
                new_obj.universal_thermal_climate_index_interpolated.iloc[:, indices]
            )

        if new_obj._universal_thermal_climate_index_calculated is not None:
            new_obj._universal_thermal_climate_index_calculated = (
                new_obj.universal_thermal_climate_index_calculated.iloc[:, indices]
            )

        return new_obj

    def calculate_comfortable_time_percentage(
        self,
        analysis_periods: tuple(AnalysisPeriod) = (AnalysisPeriod(),),
        comfort_limits: tuple[float] = (9, 26),
    ) -> pd.Series:
        """Calculate the proportion of comfortable hours for each point in the Spatial case."""
        _bool = analysis_period_to_boolean(analysis_periods)
        _filtered = self.universal_thermal_climate_index_calculated[_bool]
        _iscomfortable = (_filtered >= min(comfort_limits)) & (
            _filtered <= max(comfort_limits)
        )
        _hourscomfortable = _iscomfortable.sum(axis=0)
        _percenttimecomfortable = _hourscomfortable / _bool.sum()
        return _percenttimecomfortable

    def calculate_comfortable_area_percentage(
        self,
        time_proportion_threshold: float,
        analysis_periods: tuple[AnalysisPeriod] = (AnalysisPeriod(),),
        comfort_limits: tuple[float] = (9, 26),
    ) -> float:
        """_"""
        if time_proportion_threshold >= 1 or time_proportion_threshold <= 0:
            raise ValueError("Time proportion threshold must be between 0 and 1.")

        _percenttimecomfortable = self.calculate_comfortable_time_percentage(
            analysis_periods, comfort_limits
        )
        return (_percenttimecomfortable > time_proportion_threshold).sum() / len(
            _percenttimecomfortable
        )

    def calculate_annual_comfortable_hours(
        self,
        analysis_period_groups: tuple[tuple[AnalysisPeriod]] = ((AnalysisPeriod(),),),
        comfort_limit_groups: tuple[tuple[float]] = ((9, 26),),
    ) -> float:
        """_"""
        # given a list of analysis periods and comfort limits, calculate the annual comfortable hours
        # across all AP hours within teh comofrt limts per AP period, in the year
        if len(analysis_period_groups) != len(comfort_limit_groups):
            raise ValueError(
                "The number of analysis period groups must match the number of comfort limit groups."
            )

        annual_hours = 0
        annual_comf_hours = 0
        for ap_group, cl_group in zip(analysis_period_groups, comfort_limit_groups):
            _bool = analysis_period_to_boolean(ap_group)
            _filtered = self.universal_thermal_climate_index_calculated[_bool]
            _iscomfortable = (_filtered >= min(cl_group)) & (_filtered <= max(cl_group))
            _hourscomfortable = _iscomfortable.sum(axis=0)
            annual_comf_hours += _hourscomfortable.sum()
            annual_hours += _iscomfortable.shape[0] * _iscomfortable.shape[1]
        return annual_comf_hours / annual_hours

    def london_comfort_category(
        self,
        metric: SpatialMetric,
        comfort_limits: tuple[float] = (0, 32),
        hours: list[float] = range(8, 21, 1),
    ) -> pd.Series:
        """Calculate the London Thermal Comfort category for this Spatial case."""
        if metric.value == SpatialMetric.UTCI_INTERPOLATED.value:
            metric_values = self.universal_thermal_climate_index_interpolated
        elif metric.value == SpatialMetric.UTCI_CALCULATED.value:
            metric_values = self.universal_thermal_climate_index_calculated
        else:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )
        return rwdi_london_thermal_comfort_category(
            metric_values, comfort_limits, hours
        )

    # def direct_sun_hours(
    #     self,
    #     analysis_period: AnalysisPeriod,
    # ) -> pd.Series:
    #     """Calculate a set of direct sun hours datasets.

    #     Args:
    #         analysis_period (AnalysisPeriod, optional):
    #             An AnalysisPeriod, including timestep to simulate.

    #     Returns:
    #         pd.Series:
    #             A series containing results.
    #     """
    #     # move results files into spatial directory once completed
    #     working_dir = simulation_directory(self.model)
    #     res_dir = working_dir / "sunlight_hours"
    #     res_dir.mkdir(parents=True, exist_ok=True)
    #     results_file: Path = (
    #         res_dir
    #         / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=True)}.res"
    #     )
    #     if results_file.exists():
    #         return load_res(results_file).squeeze()

    #     result = dsh(self.model, self.simulation_result.epw, analysis_period)

    #     # make a directory in the current case to contain results files
    #     target_dir = self.spatial_simulation_directory / "sunlight_hours"
    #     target_dir.mkdir(parents=True, exist_ok=True)

    #     shutil.copy2(results_file.as_posix(), target_dir.as_posix())

    #     return result

    def get_spatial_metric(self, metric: SpatialMetric) -> pd.DataFrame:
        """Return the dataframe associated with the given metric."""
        if metric.value == SpatialMetric.POINTS.value:
            return self.points
        if metric.value == SpatialMetric.DBT_EPW.value:
            return self.dry_bulb_temperature_epw
        if metric.value == SpatialMetric.RAD_TOTAL.value:
            return self.irradiance_total
        if metric.value == SpatialMetric.RAD_DIRECT.value:
            return self.irradiance_direct
        if metric.value == SpatialMetric.RAD_DIFFUSE.value:
            return self.irradiance_diffuse
        if metric.value == SpatialMetric.SKY_VIEW.value:
            return self.sky_view
        if metric.value == SpatialMetric.UTCI_INTERPOLATED.value:
            return self.universal_thermal_climate_index_interpolated
        if metric.value == SpatialMetric.UTCI_CALCULATED.value:
            return self.universal_thermal_climate_index_calculated
        if metric.value == SpatialMetric.DBT_EPW.value:
            return self.dry_bulb_temperature_epw
        if metric.value == SpatialMetric.RH_EPW.value:
            return self.relative_humidity_epw
        if metric.value == SpatialMetric.WS_EPW.value:
            return self.wind_speed_epw
        if metric.value == SpatialMetric.WS_CFD.value:
            return self.wind_speed_cfd
        if metric.value == SpatialMetric.WD_EPW.value:
            return self.wind_direction_epw
        # if metric.value == SpatialMetric.EVAP_CLG.value:
        #     return self.evaporative_cooling_magnitude
        # if metric.value == SpatialMetric.DBT_EVAP.value:
        #     return self.dry_bulb_temperature_evap
        # if metric.value == SpatialMetric.RH_EVAP.value:
        #     return self.relative_humidity_evap
        if metric.value == SpatialMetric.MRT_INTERPOLATED.value:
            return self.mean_radiant_temperature_interpolated
        raise ValueError(f"{metric} cannot be obtained!")

    def plot_typical_point_in_time(
        self, metric: SpatialMetric, month: int, hour: int, levels: list[float] = None
    ) -> plt.Figure:
        """Create a typical point-in-time plot of the given metric."""

        # test that metric passed is temporal
        if not metric.is_temporal():
            raise ValueError(f"{metric} is not a temporal metric.")

        if not month in range(1, 13, 1):
            raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
        if not hour in range(0, 24, 1):
            raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

        df = self.get_spatial_metric(metric)

        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting {metric.description()} for {calendar.month_abbr[month]} {hour:02d}:00"
        )

        # get global levels for this metric (to enable like-for-like comparison between time periods.)
        tcf_properties = metric.tricontourf_kwargs()
        if levels is not None:
            tcf_properties["levels"] = levels
        tc_properties = metric.tricontour_kwargs()

        z = (
            df.groupby(
                [
                    df.index.month,
                    df.index.hour,
                ],
                axis=0,
            )
            .mean()
            .loc[month, hour]
        ).values

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # add contour-fill
        tcf = ax.tricontourf(self._triangulation, z, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(metric.description())

        # add contour lines
        if len(tc_properties["levels"]) != 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tcs = ax.tricontour(self._triangulation, z, **tc_properties)
                ax.clabel(tcs, inline=1, fontsize="small", colors=["k"])
                cbar.add_lines(tcs)

        # add title
        ax.set_title(
            f"Typical {metric.description()}\n{calendar.month_abbr[month]} {hour:02d}:00",
            ha="left",
            va="bottom",
            x=0,
        )

        plt.tight_layout()

        return fig

    def plot_sky_view(self) -> plt.Figure:
        """Return a sky-view plot figure object."""

        metric = SpatialMetric.SKY_VIEW
        CONSOLE_LOGGER.info(f"[{self}] - Plotting {metric.description()}")
        values = self.sky_view.squeeze().values

        tcf_properties = metric.tricontourf_kwargs()
        tc_properties = metric.tricontour_kwargs()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # add contour-fill
        tcf = ax.tricontourf(self._triangulation, values, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(metric.description())

        # add contour lines if present
        if len(tc_properties["levels"]) != 0:
            tcs = ax.tricontour(self._triangulation, values, **tc_properties)
            ax.clabel(tcs, inline=1, fontsize="small", colors=["k"])
            cbar.add_lines(tcs)

        # add title
        ax.set_title(
            f"Proportion of sky visible\nAverage: {np.mean(values/100):0.1%}",
            ha="left",
            va="bottom",
            x=0,
        )

        plt.tight_layout()

        return fig

    def plot_london_comfort_category(
        self,
        metric: SpatialMetric,
        comfort_limits: tuple[float] = (0, 32),
        hours: list[float] = range(8, 21, 1),
    ) -> plt.Figure:
        """Return a plot showing the RWDI London thermal comfort category."""

        comfort_category = self.london_comfort_category(metric, comfort_limits, hours)

        # convert categori

        CONSOLE_LOGGER.info(f"[{self}] - Plotting London Thermal Comfort Category")

        # convert category to numeric
        mapper = {
            "Transient": 0,
            "Short-term Seasonal": 1,
            "Short-term": 2,
            "Seasonal": 3,
            "All Season": 4,
        }
        values = comfort_category.replace(mapper)

        # plot
        tcf_properties = {
            "cmap": ListedColormap(
                [
                    "#DE2E26",
                    "#FAB92D",
                    "#1eFFFF",
                    "#C86eBE",
                    "#378c4b",
                ]
            ),
            "levels": np.arange(-0.5, 4.6, 1),
            "extend": "neither",
        }

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # add contour-fill
        tcf = ax.tricontourf(self._triangulation, values, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label("London Thermal Comfort Category")
        cbar.set_ticks(list(mapper.values()))
        cbar.set_ticklabels([i.replace(" ", "\n") for i in mapper])

        # add title
        ax.set_title(
            'Thermal Comfort Category\nFrom "Thermal Comfort Guidelines for developments in the City of London"',
            ha="left",
            va="bottom",
            x=0,
        )

        plt.tight_layout()

        return fig

    def plot_wind_average(self, analysis_period: AnalysisPeriod) -> plt.Figure:
        """Return a figure showing typical wind-speed for the given analysis period."""

        metric: SpatialMetric = SpatialMetric.WS_CFD
        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting {metric.description()} for {describe_analysis_period(analysis_period)}"
        )
        values = (
            self.get_spatial_metric(metric)
            .iloc[list(analysis_period.hoys_int)]
            .mean()
            .values
        )

        tcf_properties = metric.tricontourf_kwargs()
        tc_properties = metric.tricontour_kwargs()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # add contour-fill
        tcf = ax.tricontourf(self._triangulation, values, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(metric.description())

        # add contour lines if present
        if len(tc_properties["levels"]) != 0:
            tcs = ax.tricontour(self._triangulation, values, **tc_properties)
            ax.clabel(tcs, inline=1, fontsize="small", colors=["k"])
            cbar.add_lines(tcs)

        # add title
        ax.set_title(
            f"{describe_analysis_period(analysis_period)}\nAverage {metric.description()}\nAverage: {values.mean():0.1f}m/s",
            ha="left",
            va="bottom",
            x=0,
        )

        plt.tight_layout()

        return fig

    def plot_mrt_average(self, analysis_period: AnalysisPeriod) -> plt.Figure:
        """Return a figure showing typical mrt for the given analysis period."""

        metric: SpatialMetric = SpatialMetric.MRT_INTERPOLATED
        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting {metric.description()} for {describe_analysis_period(analysis_period)}"
        )

        # obtain levels
        low = self.get_spatial_metric(metric).min().min()
        high = self.get_spatial_metric(metric).max().max()
        levels = np.linspace(low, high, 51)
        values = (
            self.get_spatial_metric(metric)
            .iloc[list(analysis_period.hoys_int)]
            .mean()
            .values
        )

        tcf_properties = metric.tricontourf_kwargs()
        tc_properties = metric.tricontour_kwargs()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # add contour-fill
        tcf = ax.tricontourf(
            self._triangulation, values, levels=levels, **tcf_properties
        )

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(tcf, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_label(metric.description())

        # add contour lines if present
        if len(tc_properties["levels"]) != 0:
            tcs = ax.tricontour(self._triangulation, values, **tc_properties)
            ax.clabel(tcs, inline=1, fontsize="small", colors=["k"])
            cbar.add_lines(tcs)

        # add title
        ax.set_title(
            f"{describe_analysis_period(analysis_period)}\nAverage {metric.description()}\nAverage: {values.mean():0.1f}C",
            ha="left",
            va="bottom",
            x=0,
        )

        plt.tight_layout()

        return fig

    # def plot_direct_sun_hours(self, analysis_period: AnalysisPeriod) -> plt.Figure:
    #     """Plot the  direct-sun-hours for the given analysis period."""

    #     metric = SpatialMetric.DIRECT_SUN_HOURS
    #     CONSOLE_LOGGER.info(
    #         f"[{self}] - Plotting {metric.description()} for {describe_analysis_period(analysis_period, include_timestep=True)}"
    #     )
    #     series = self.direct_sun_hours(analysis_period)
    #     tcf_properties = metric.tricontourf_kwargs()
    #     tc_properties = metric.tricontour_kwargs()

    #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #     ax.set_aspect("equal")
    #     ax.axis("off")
    #     ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
    #     ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

    #     # add contour-fill
    #     tcf = ax.tricontourf(self._triangulation, series.values, **tcf_properties)

    #     # plot colorbar
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
    #     cbar = plt.colorbar(tcf, cax=cax)
    #     cbar.outline.set_visible(False)
    #     cbar.set_label(metric.description())

    #     # add contour lines if present
    #     if len(tc_properties["levels"]) != 0:
    #         tcs = ax.tricontour(self._triangulation, series.values, **tc_properties)
    #         ax.clabel(tcs, inline=1, fontsize="x-small", colors=["k"])
    #         cbar.add_lines(tcs)

    #     # add title
    #     ax.set_title(
    #         f"{describe_analysis_period(analysis_period, include_timestep=False)}\n{metric.description()}\nAverage: {np.mean(series.values):0.1f} hours",
    #         ha="left",
    #         va="bottom",
    #         x=0,
    #     )

    #     plt.tight_layout()

    #     return fig

    def plot_sunpath(self, analysis_period: AnalysisPeriod) -> plt.Figure:
        """Return a figure showing the sunpath for this simulation at the given time."""
        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting Sunpath for {describe_analysis_period(analysis_period, include_timestep=True)}"
        )
        fig = sunpath(
            self.simulation_result.epw,
            analysis_period,
            show_legend=False,
            show_title=False,
            sun_size=20,
        )
        return fig

    def plot_hours_comfortable(
        self,
        analysis_period: AnalysisPeriod,
        metric: SpatialMetric,
        comfort_thresholds: tuple[float] = (9, 26),
    ) -> plt.Figure:
        """Return a figure showing an "hours comfortable" plot."""

        if metric.value == SpatialMetric.UTCI_INTERPOLATED.value:
            metric_values = self.universal_thermal_climate_index_interpolated
        elif metric.value == SpatialMetric.UTCI_CALCULATED.value:
            metric_values = self.universal_thermal_climate_index_calculated
        else:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )

        if len(comfort_thresholds) != 2:
            raise ValueError("comfort_limits must be a list of two values.")

        if analysis_period.timestep != 1:
            warnings.warn(
                "Hours Comfortable can only be calculated for hourly values (for now). AnalysisPeriod will updated to reflect this."
            )
            analysis_period = AnalysisPeriod(
                st_month=analysis_period.st_month,
                st_day=analysis_period.st_day,
                st_hour=analysis_period.st_hour,
                end_month=analysis_period.end_month,
                end_day=analysis_period.end_day,
                end_hour=analysis_period.end_hour,
            )

        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting Hours Comfortable for {describe_analysis_period(analysis_period)}"
        )

        z_temp = metric_values.iloc[list(analysis_period.hoys_int), :]
        z = (
            ((z_temp >= min(comfort_thresholds)) & (z_temp <= max(comfort_thresholds)))
            .sum()
            .values
        ) / len(analysis_period.hoys_int)
        z_mean = np.mean(z)

        # set tricontour properties
        tcf_properties = {
            "cmap": "Greens",
            "levels": np.linspace(0, 1, 101),
            "extend": "both",
        }
        tcl_properties = {
            "colors": "k",
            "linestyles": "-",
            "linewidths": 0.5,
            "alpha": 0.5,
            "levels": [0.25, 0.5, 0.75],
        }

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # plot tricontour of z(comfortable hours) values
        tcf = ax.tricontourf(self._triangulation, z, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(
            tcf,
            cax=cax,
            format=PercentFormatter(xmax=1),
        )
        cbar.outline.set_visible(False)
        cbar.set_label("Hours comfortable")

        # add contour-lines (if any are present)
        if not (all(z < 0.5) or all(z > 0.5)):
            tcl = ax.tricontour(self._triangulation, z, **tcl_properties)
            ax.clabel(
                tcl,
                inline=1,
                fontsize="small",
                fmt=StrMethodFormatter("{x:0.0%}"),
            )
            cbar.add_lines(tcl)

        # title
        ax.set_title(
            f'{describe_analysis_period(analysis_period)} ({len(analysis_period.hoys_int)} hours)\n"Comfortable" between {min(comfort_thresholds)} and {max(comfort_thresholds)}°C UTCI\nAverage: {z_mean:0.1%}'
        )

        plt.tight_layout()

        return fig

    def plot_hours_cold(
        self,
        analysis_period: AnalysisPeriod,
        metric: SpatialMetric,
        cold_threshold: float = 9,
    ) -> plt.Figure:
        """Return a figure showing an "hours cold" plot."""

        if metric.value == SpatialMetric.UTCI_INTERPOLATED.value:
            metric_values = self.universal_thermal_climate_index_interpolated
        elif metric.value == SpatialMetric.UTCI_CALCULATED.value:
            metric_values = self.universal_thermal_climate_index_calculated
        else:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )

        if analysis_period.timestep != 1:
            warnings.warn(
                "Hours Cold can only be calculated for hourly values (for now). AnalysisPeriod will updated to reflect this."
            )
            analysis_period = AnalysisPeriod(
                st_month=analysis_period.st_month,
                st_day=analysis_period.st_day,
                st_hour=analysis_period.st_hour,
                end_month=analysis_period.end_month,
                end_day=analysis_period.end_day,
                end_hour=analysis_period.end_hour,
            )

        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting Hours Cold for {describe_analysis_period(analysis_period)}"
        )

        z_temp = metric_values.iloc[list(analysis_period.hoys_int), :]
        z = (z_temp < cold_threshold).sum().values / len(analysis_period.hoys_int)
        z_mean = np.mean(z)

        # set tricontour properties
        tcf_properties = {
            "cmap": "Blues",
            "levels": np.linspace(0, 1, 101),
            "extend": "both",
        }
        tcl_properties = {
            "colors": "k",
            "linestyles": "-",
            "linewidths": 0.5,
            "alpha": 0.5,
            "levels": [0.25, 0.5, 0.75],
        }

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # plot tricontour of z(cold hours) values
        tcf = ax.tricontourf(self._triangulation, z, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(
            tcf,
            cax=cax,
            format=PercentFormatter(xmax=1),
        )
        cbar.outline.set_visible(False)
        cbar.set_label("Hours cold")

        # add contour-lines (if any are present)
        if not (all(z < 0.5) or all(z > 0.5)):
            tcl = ax.tricontour(self._triangulation, z, **tcl_properties)
            ax.clabel(
                tcl,
                inline=1,
                fontsize="small",
                fmt=StrMethodFormatter("{x:0.0%}"),
            )
            cbar.add_lines(tcl)

        # title
        ax.set_title(
            f'{describe_analysis_period(analysis_period)} ({len(analysis_period.hoys_int)} hours)\n"Cold" below {cold_threshold}°C UTCI\nAverage: {z_mean:0.1%}'
        )

        plt.tight_layout()

        return fig

    def plot_hours_hot(
        self,
        analysis_period: AnalysisPeriod,
        metric: SpatialMetric,
        hot_threshold: float = 26,
    ) -> plt.Figure:
        """Return a figure showing an "hours hot" plot."""

        if metric.value == SpatialMetric.UTCI_INTERPOLATED.value:
            metric_values = self.universal_thermal_climate_index_interpolated
        elif metric.value == SpatialMetric.UTCI_CALCULATED.value:
            metric_values = self.universal_thermal_climate_index_calculated
        else:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )

        if analysis_period.timestep != 1:
            warnings.warn(
                "Hours Hot can only be calculated for hourly values (for now). AnalysisPeriod will updated to reflect this."
            )
            analysis_period = AnalysisPeriod(
                st_month=analysis_period.st_month,
                st_day=analysis_period.st_day,
                st_hour=analysis_period.st_hour,
                end_month=analysis_period.end_month,
                end_day=analysis_period.end_day,
                end_hour=analysis_period.end_hour,
            )

        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting Hours Hot for {describe_analysis_period(analysis_period)}"
        )

        z_temp = metric_values.iloc[list(analysis_period.hoys_int), :]
        z = (z_temp > hot_threshold).sum().values / len(analysis_period.hoys_int)
        z_mean = np.mean(z)

        # set tricontour properties
        tcf_properties = {
            "cmap": "Reds",
            "levels": np.linspace(0, 1, 101),
            "extend": "both",
        }
        tcl_properties = {
            "colors": "k",
            "linestyles": "-",
            "linewidths": 0.5,
            "alpha": 0.5,
            "levels": [0.25, 0.5, 0.75],
        }

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim([min(self.points.x.values), max(self.points.x.values)])
        ax.set_ylim([min(self.points.y.values), max(self.points.y.values)])

        # plot tricontour of z(cold hours) values
        tcf = ax.tricontourf(self._triangulation, z, **tcf_properties)

        # plot colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
        cbar = plt.colorbar(
            tcf,
            cax=cax,
            format=PercentFormatter(xmax=1),
        )
        cbar.outline.set_visible(False)
        cbar.set_label("Hours hot")

        # add contour-lines (if any are present)
        if not (all(z < 0.5) or all(z > 0.5)):
            tcl = ax.tricontour(self._triangulation, z, **tcl_properties)
            ax.clabel(
                tcl,
                inline=1,
                fontsize="small",
                fmt=StrMethodFormatter("{x:0.0%}"),
            )
            cbar.add_lines(tcl)

        # title
        ax.set_title(
            f'{describe_analysis_period(analysis_period)} ({len(analysis_period.hoys_int)} hours)\n"Hot" above {hot_threshold}°C UTCI\nAverage: {z_mean:0.1%}'
        )

        plt.tight_layout()

        return fig

    def plot_hours_combined(
        self,
        analysis_period: AnalysisPeriod,
        metric: SpatialMetric,
        comfort_thresholds: tuple[float] = (0, 28),
    ) -> plt.Figure:
        """Return an Image object containing a combination of "hours cold/hot/comfortable"."""

        # # create figures
        # cold_fig = self.plot_hours_cold(
        #     analysis_period, metric, cold_threshold=min(comfort_thresholds)
        # )
        # comfortable_fig = self.plot_hours_comfortable(
        #     analysis_period, metric, comfort_thresholds=comfort_thresholds
        # )
        # hot_fig = self.plot_hours_hot(
        #     analysis_period, metric, hot_threshold=max(comfort_thresholds)
        # )

        # # convert to image objects
        # cold_im = Image.frombytes(
        #     "RGBA", cold_fig.canvas.get_width_height(), cold_fig.canvas.tostring_argb()
        # )
        # comfortable_im = Image.frombytes(
        #     "RGBA",
        #     comfortable_fig.canvas.get_width_height(),
        #     comfortable_fig.canvas.tostring_argb(),
        # )
        # hot_im = Image.frombytes(
        #     "RGBA", hot_fig.canvas.get_width_height(), hot_fig.canvas.tostring_argb()
        # )

        # # combine images in single image object
        # images = [
        #     cold_im,
        #     comfortable_im,
        #     hot_im,
        # ]
        # widths, heights = zip(*(i.size for i in images))
        # total_width = sum(widths)
        # max_height = max(heights)
        # new_im = Image.new("RGBA", (total_width, max_height))
        # x_offset = 0
        # for im in images:
        #     new_im.paste(im, (x_offset, 0))
        #     x_offset += im.size[0]

        # plt.close(cold_fig)
        # plt.close(comfortable_fig)
        # plt.close(hot_fig)

        # return new_im

        raise NotImplementedError("Not yet working!")

    def plot_spatial_point_locations(self, n: int = 100) -> plt.Figure:
        """Return the spatial point locations figure."""

        CONSOLE_LOGGER.info(
            f"[{self}] - Plotting spatial point locations, every {n}th point."
        )

        x = self.points.x.values
        y = self.points.y.values
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set_aspect("equal")
        ax.scatter(x, y, c="#555555", s=1)
        for i in np.arange(0, len(self.points.x.values), n):
            ax.scatter(x[i], y[i], c="red", s=2)
            ax.text(
                x[i],
                y[i],
                i,
                ha="center",
                va="center",
                fontsize="x-small",
            )
        plt.tight_layout()

        return fig

    def plot_sky_view_pov(
        self,
        point_index: int,
        point_identifier: str,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        show_sunpath: bool = True,
        show_skymatrix: bool = True,
    ) -> Path:
        """Generate a plot combining skymatrix, sunpath amnd contextual geometry for a given point.

        Args:
            point_index (int):
                The index of the point to plot.
            point_identifier (str):
                The identifier of the point to plot.
            analysis_period (AnalysisPeriod, optional):
                The analysis period to use for the plot. Defaults to AnalysisPeriod().
            show_sunpath (bool, optional):
                Whether to show the sunpath. Defaults to True.
            show_skymatrix (bool, optional):
                Whether to show the skymatrix. Defaults to True.

        Returns:
            Path: The path to the generated plot.
        """

        # img = sky_view_pov(
        #     model=Model.from_hbjson(
        #         list(self.spatial_simulation_directory.glob("**/*.hbjson"))[0]
        #     ),
        #     sensor=Point3D.from_array(
        #         self.points.iloc[point_index][["x", "y", "z"]].values
        #     ),
        #     epw=self.simulation_result.epw,
        #     analysis_period=AnalysisPeriod(timestep=5),
        #     cmap=None,
        #     norm=None,
        #     data_collection=None,
        #     density=4,
        #     show_sunpath=show_sunpath,
        #     show_skymatrix=show_skymatrix,
        #     title=f"{describe_loc(self.simulation_result.epw.location)}\n{self.spatial_simulation_directory.stem}\n{describe_analysis_period(analysis_period)}\n{point_identifier}",
        # )
        # save_path = self._plot_directory / f"point_{point_identifier}_skyview.png"
        # img.save(save_path, dpi=(500, 500))
        # return save_path

        raise NotImplementedError("Not yet working!")

    def plot_hottest_coldest_days(self) -> plt.Figure:
        """Return a figure showing the hottest and coldest days."""

        date_form = DateFormatter("%b-%d")

        fig, ax = plt.subplots(1, 1, figsize=(16, 2))
        collection_to_series(self.simulation_result.epw.dry_bulb_temperature).resample(
            "D"
        ).mean().plot(ax=ax, c="orange")
        lims = ax.get_ylim()
        ax.axvline(self.hottest_day, c="red")
        ax.text(
            self.hottest_day,
            lims[1],
            f"Hottest day ({self.hottest_day:%b %d})",
            ha="left",
            va="bottom",
        )
        ax.axvline(self.coldest_day, c="blue")
        ax.text(
            self.coldest_day,
            lims[1],
            f"Coldest day ({self.coldest_day:%b %d})",
            ha="left",
            va="bottom",
        )
        ax.set_ylabel("Daily mean\nDry bulb temperature (C)")
        ax.xaxis.set_major_formatter(date_form)

        plt.tight_layout()
        return fig

    def summarise_point(
        self,
        point_index: int,
        point_identifier: str,
        metric: SpatialMetric,
        comfort_limits: tuple = (9, 26),
    ) -> None:
        """Return the figure object showing the location of a specific point."""

        if metric.value == SpatialMetric.UTCI_INTERPOLATED.value:
            metric_values = self.universal_thermal_climate_index_interpolated
        elif metric.value == SpatialMetric.UTCI_CALCULATED.value:
            metric_values = self.universal_thermal_climate_index_calculated
        else:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )

        # create the datacollection for the given point index
        point_utci = collection_from_series(
            metric_values.iloc[:, point_index].rename(
                "Universal Thermal Climate Index (C)"
            )
        )

        # plot point location
        save_path = self._plot_directory / f"point_{point_identifier}_location.png"
        if not save_path.exists():
            CONSOLE_LOGGER.info(
                f"[{self}] - Plotting point location for {point_identifier}"
            )
            x = self.points.x.values
            y = self.points.y.values
            xlims = [min(x), max(x)]
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            ax.set_aspect("equal")
            ax.axis("off")
            ax.tricontourf(
                self._triangulation,
                self.irradiance_total.mean(axis=0),
                levels=100,
                cmap="bone",
            )
            # ax.scatter(x, y, c="#555555", s=0.1)
            pt_size = (xlims[1] - xlims[0]) / 3
            ax.scatter(x[point_index], y[point_index], s=pt_size, c="red")
            ax.text(
                x[point_index] + (pt_size / 10),
                y[point_index],
                point_identifier,
                ha="left",
                va="center",
                fontsize="large",
            )
            plt.tight_layout()
            fig.savefig(save_path, transparent=True)

        # create openfield UTCI plot
        save_path = self._plot_directory / "point_Openfield_utci.png"
        if not save_path.exists():
            CONSOLE_LOGGER.info(f"[{self}] - Plotting Openfield UTCI")
            fig = utci_heatmap_histogram(self._unshaded_utci, "Openfield")
            fig.savefig(save_path, transparent=True, bbox_inches="tight")

        # # create openfield UTCI distance to comfortable plot
        # save_path = self._plot_directory / "point_Openfield_distance_to_comfortable.png"
        # if not save_path.exists():
        #     CONSOLE_LOGGER.info(
        #         f"[{self}] - Plotting Openfield UTCI distance to comfortable"
        #     )
        #     fig = utci_distance_to_comfortable(self._unshaded_utci, "Openfield")
        #     fig.savefig(save_path, transparent=True, bbox_inches="tight")

        # create point location UTCI plot
        save_path = (
            self._plot_directory / f"point_{sanitise_string(point_identifier)}_utci.png"
        )
        if not save_path.exists():
            CONSOLE_LOGGER.info(f"[{self}] - Plotting {point_identifier} UTCI")
            f = utci_heatmap_histogram(point_utci, f"{self} - {point_identifier}")
            f.savefig(save_path, transparent=True, bbox_inches="tight")

        # # create pt location UTCI distance to comfortable plot
        # save_path = (
        #     self._plot_directory
        #     / f"point_{sanitise_string(point_identifier)}_distance_to_comfortable.png"
        # )
        # if not save_path.exists():
        #     CONSOLE_LOGGER.info(
        #         f"[{self}] - Plotting {point_identifier} UTCI distance to comfortable"
        #     )
        #     f = utci_distance_to_comfortable(point_utci, f"{self} - {point_identifier}")
        #     f.savefig(save_path, transparent=True, bbox_inches="tight")

        # create pt location UTCI difference
        save_path = (
            self._plot_directory
            / f"point_{sanitise_string(point_identifier)}_difference.png"
        )
        if not save_path.exists():
            CONSOLE_LOGGER.info(
                f"[{self}] - Plotting difference between Openfield and {point_identifier} UTCI"
            )
            f = utci_heatmap_difference(
                self._unshaded_utci,
                point_utci,
                f"{self} - Difference between Openfield UTCI and {point_identifier} UTCI",
            )
            f.savefig(save_path, transparent=True, bbox_inches="tight")

        # # create pt location CSV UTCI simplified
        # save_path = (
        #     self._plot_directory
        #     / f"point_{sanitise_string(point_identifier)}_monthlysummary.png"
        # )
        # if not save_path.exists():
        #     pt_monthly_utci = describe_utci_monthly(
        #         utci_collection=point_utci,
        #         density=True,
        #         comfort_limits=comfort_limits,
        #         analysis_periods=analysis_period,
        #     )
        #     pt_monthly_utci.to_csv(save_path)

        # create pt location sky view
        # self.plot_sky_view_pov(point_index, point_identifier, Vector3D(0, 0, 0.5))

        plt.close("all")

    def summarise_point_from_json(
        self,
        metric: SpatialMetric,
        comfort_limits: tuple = (9, 26),
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> None:
        """For each point given in a JSON file, summarise the point."""

        focus_pts_file = self.spatial_simulation_directory / "focus_points.json"
        if not focus_pts_file.exists():
            warnings.warn(
                f"A focus_points.json file does not exist in {self.spatial_simulation_directory}."
            )
            return None

        with open(focus_pts_file, "r", encoding="utf-8") as fp:
            focus_pts = eval(fp.read())  # pylint: disable=eval-used

        for point_identifier, point_index in focus_pts.items():
            self.summarise_point(
                point_index, point_identifier, metric, comfort_limits, analysis_period
            )

        return None

    def run_all(
        self,
        pt_locations: bool = False,
        sky_view: bool = False,
        sunpaths: bool = False,
        typical_wind: bool = False,
        typical_mrt: bool = False,
        comfort_percentages: bool = False,
        sunlight_hours: bool = False,
        london_comfort: bool = False,
        comfort_limits: tuple[float] = (9, 26),
    ) -> None:
        """
        Run all plotting methods

        Args:
            pt_locations (bool, optional):
                Plot point locations. Defaults to False.
            sky_view (bool, optional):
                Plot sky view. Defaults to False.
            sunpaths (bool, optional):
                Plot sunpaths. Defaults to False.
            typical_wind (bool, optional):
                Plot typical wind. Defaults to False.
            typical_mrt (bool, optional):
                Plot typical MRT. Defaults to False.
            comfort_percentages (bool, optional):
                Plot comfort percentages. Defaults to False.
            sunlight_hours (bool, optional):
                Plot sunlight hours. Defaults to False.
            london_comfort (bool, optional):
                Plot London comfort. Defaults to False.
            comfort_limits (tuple[float], optional):
                Comfort limits. Defaults to (9, 26).

        """

        cold_threshold = min(comfort_limits)
        hot_threshold = max(comfort_limits)

        ################
        # Non-temporal #
        ################

        if pt_locations:
            save_path = self._plot_directory / "pt_locations.png"
            if not save_path.exists():
                fig = self.plot_spatial_point_locations()
                fig.savefig(save_path, transparent=True, bbox_inches="tight")
                plt.close(fig)

        if sky_view:
            save_path = self._plot_directory / "sky_view.png"
            if not save_path.exists():
                fig = self.plot_sky_view()
                fig.savefig(save_path, transparent=True, bbox_inches="tight")
                plt.close(fig)

        ############
        # Temporal #
        ############

        for analysis_period in self.analysis_periods:
            if sunpaths:
                save_path = (
                    self._plot_directory
                    / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=True)}_sunpath.png"
                )
                if not save_path.exists():
                    fig = self.plot_sunpath(analysis_period)
                    fig.savefig(save_path, transparent=True, bbox_inches="tight")
                    plt.close(fig)

            if typical_mrt:
                if analysis_period.timestep == 1:
                    save_path = (
                        self._plot_directory
                        / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=False)}_typical_mrt.png"
                    )
                    if not save_path.exists():
                        fig = self.plot_mrt_average(analysis_period)
                        fig.savefig(save_path, transparent=True, bbox_inches="tight")
                        plt.close(fig)

            if typical_wind:
                # pylint disable=broad-exception-caught
                try:
                    if analysis_period.timestep == 1:
                        save_path = (
                            self._plot_directory
                            / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=False)}_typical_wind.png"
                        )
                        if not save_path.exists():
                            fig = self.plot_wind_average(analysis_period)
                            fig.savefig(
                                save_path, transparent=True, bbox_inches="tight"
                            )
                            plt.close(fig)
                except Exception:
                    pass
                # pylint enable=broad-exception-caught

            if comfort_percentages:
                try:
                    if analysis_period.timestep == 1:
                        # time cold
                        save_path = (
                            self._plot_directory
                            / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=False)}_hours_cold.png"
                        )
                        if not save_path.exists():
                            fig = self.plot_hours_cold(
                                analysis_period,
                                SpatialMetric.UTCI_CALCULATED,
                                cold_threshold=cold_threshold,
                            )
                            fig.savefig(
                                save_path, transparent=True, bbox_inches="tight"
                            )
                            plt.close(fig)

                        # time comfortable
                        save_path = (
                            self._plot_directory
                            / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=False)}_hours_comfortable.png"
                        )
                        if not save_path.exists():
                            fig = self.plot_hours_comfortable(
                                analysis_period,
                                SpatialMetric.UTCI_CALCULATED,
                                comfort_thresholds=(cold_threshold, hot_threshold),
                            )
                            fig.savefig(
                                save_path, transparent=True, bbox_inches="tight"
                            )
                            plt.close(fig)

                        # time hot
                        save_path = (
                            self._plot_directory
                            / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=False)}_hours_hot.png"
                        )
                        if not save_path.exists():
                            fig = self.plot_hours_hot(
                                analysis_period,
                                SpatialMetric.UTCI_CALCULATED,
                                hot_threshold=hot_threshold,
                            )
                            fig.savefig(
                                save_path, transparent=True, bbox_inches="tight"
                            )
                            plt.close(fig)
                except Exception:
                    pass

            datetimes = analysis_period_to_datetimes(analysis_period)
            time_delta = datetimes.max() - datetimes.min()

            if time_delta <= pd.Timedelta(days=1) and analysis_period.timestep > 1:
                pass
                # if sunlight_hours:
                #     save_path = (
                #         self._plot_directory
                #         / f"{describe_analysis_period(analysis_period, save_path=True, include_timestep=True)}_direct_sun_hours.png"
                #     )
                #     if not save_path.exists():
                #         fig = self.plot_direct_sun_hours(analysis_period)
                #         fig.savefig(save_path, transparent=True, bbox_inches="tight")
                #         plt.close(fig)

        if london_comfort:
            try:
                save_path = self._plot_directory / "LondonThermalComfort.png"
                if not save_path.exists():
                    fig = self.plot_london_comfort_category(
                        metric=SpatialMetric.UTCI_CALCULATED
                    )
                    fig.savefig(save_path, transparent=True, bbox_inches="tight")
                    plt.close(fig)
            except Exception:
                pass


def spatial_comfort_possible(sim_dir: Path) -> bool:
    """Checks whether spatial_comfort processing is possible for a given simulation_directory.

    Args:
        sim_dir (Path):
            A folder containing Honeybee-Radiance Sky-View and Annual Irradiance results.

    Returns:
        bool:
            True if possible. If impossible, then an error is raised instead.
    """

    sim_dir = Path(sim_dir)

    # if process already run (and output files already exist), then return true
    if all(
        [
            SpatialMetric.RAD_TOTAL.filepath(sim_dir).exists(),
            SpatialMetric.SKY_VIEW.filepath(sim_dir).exists(),
        ]
    ):
        return True

    # Check for annual irradiance data
    annual_irradiance_directory = sim_dir / "annual_irradiance"
    if (
        not annual_irradiance_directory.exists()
        or len(list((annual_irradiance_directory / "results").glob("**/*.ill"))) == 0
    ):
        raise FileNotFoundError(
            f"Annual-irradiance data is not available in {annual_irradiance_directory}."
        )

    # Check for sky-view data
    sky_view_directory = sim_dir / "sky_view"
    if not (sky_view_directory).exists():
        raise FileNotFoundError(
            f"Sky-view data is not available in {sky_view_directory}."
        )
    res_files = list((sky_view_directory / "results" / "sky_view").glob("**/*.res"))
    if len(res_files) == 0:
        res_files += list((sky_view_directory / "results").glob("**/*.res"))
    if len(res_files) == 0:
        raise FileNotFoundError(
            f"Sky-view data is not available in {sky_view_directory}."
        )
    if len(res_files) != 1:
        raise ValueError(
            "This process is currently only possible for a single Analysis Grid - multiple files found."
        )

    return True
