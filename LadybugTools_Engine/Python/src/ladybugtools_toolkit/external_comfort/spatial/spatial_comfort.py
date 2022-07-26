from __future__ import annotations

import calendar
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from cached_property import cached_property
from ladybug.analysisperiod import AnalysisPeriod
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import (
    SimulationResult,
)
from ladybugtools_toolkit.external_comfort.spatial.calculate_mean_radiant_temperature_interpolated import (
    calculate_mean_radiant_temperature_interpolated,
)
from ladybugtools_toolkit.external_comfort.spatial.load_diffuse_irradiance import (
    load_diffuse_irradiance,
)
from ladybugtools_toolkit.external_comfort.spatial.load_direct_irradiance import (
    load_direct_irradiance,
)
from ladybugtools_toolkit.external_comfort.spatial.load_points import load_points
from ladybugtools_toolkit.external_comfort.spatial.load_sky_view import load_sky_view
from ladybugtools_toolkit.external_comfort.spatial.load_total_irradiance import (
    load_total_irradiance,
)
from ladybugtools_toolkit.external_comfort.spatial.load_universal_thermal_climate_index_interpolated import (
    load_universal_thermal_climate_index_interpolated,
)
from ladybugtools_toolkit.external_comfort.spatial.spatial_comfort_possible import (
    spatial_comfort_possible,
)
from ladybugtools_toolkit.external_comfort.spatial.temporospatial_metric import (
    TemporospatialMetric,
)
from ladybugtools_toolkit.external_comfort.thermal_comfort.utci.utci import utci
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
    from_series,
)
from ladybugtools_toolkit.plot.colormap_sequential import colormap_sequential
from ladybugtools_toolkit.plot.colormaps import (
    UTCI_BOUNDARYNORM,
    UTCI_COLORMAP,
    UTCI_LEVELS,
)
from ladybugtools_toolkit.plot.create_triangulation import create_triangulation
from ladybugtools_toolkit.plot.spatial_heatmap import spatial_heatmap
from ladybugtools_toolkit.plot.utci_distance_to_comfortable import (
    utci_distance_to_comfortable,
)
from ladybugtools_toolkit.plot.utci_heatmap_difference import utci_heatmap_difference
from ladybugtools_toolkit.plot.utci_heatmap_histogram import utci_heatmap_histogram
from matplotlib import pyplot as plt

PLOT_DPI = 300


class SpatialComfort:
    def __init__(
        self, spatial_simulation_directory: Path, simulation_result: SimulationResult
    ) -> SpatialComfort:
        """A SpatialComfort object, used to calculate spatial UTCI.

        Args:
            spatial_simulation_directory (Path): A directory path containing SkyView and Annual
                Irradiance simulation results.
            simulation_result (SimulationResult): A results object containing
                pre-simulated MRT values.

        Returns:
            SpatialComfort: A SpatialComfort object.
        """

        self.spatial_simulation_directory = Path(spatial_simulation_directory)
        self.simulation_result = simulation_result

        # check that spatial-comfort is possible for given simulation_directory
        spatial_comfort_possible(self.spatial_simulation_directory)

        # calculate baseline UTCI for shaded/unshaded
        self._unshaded_utci = utci(
            self.simulation_result.epw.dry_bulb_temperature,
            self.simulation_result.epw.relative_humidity,
            self.simulation_result.unshaded_mean_radiant_temperature,
            self.simulation_result.epw.wind_speed,
        )
        self._shaded_utci = utci(
            self.simulation_result.epw.dry_bulb_temperature,
            self.simulation_result.epw.relative_humidity,
            self.simulation_result.shaded_mean_radiant_temperature,
            self.simulation_result.epw.wind_speed,
        )

        # create directory for associated plots
        self._plot_directory = self.spatial_simulation_directory / "plots"
        self._plot_directory.mkdir(exist_ok=True, parents=True)

        # create default triangulation
        self._points_x = self.points.droplevel(0, axis=1).x.values
        self._points_y = self.points.droplevel(0, axis=1).y.values
        self._triangulation = create_triangulation(
            self._points_x,
            self._points_y,
            alpha=1.1,
            max_iterations=250,
            increment=0.01,
        )

    def _get_temporospatial_metric(self, metric: TemporospatialMetric) -> pd.DataFrame:
        """A helper method to access a temporospatial metric."""
        if metric == TemporospatialMetric.RAD_TOTAL:
            return self.total_rad
        elif metric == TemporospatialMetric.RAD_DIRECT:
            return self.direct_rad
        elif metric == TemporospatialMetric.RAD_DIFFUSE:
            return self.diffuse_rad
        elif metric == TemporospatialMetric.MRT:
            return self.mean_radiant_temperature
        elif metric == TemporospatialMetric.UTCI_CALCULATED:
            return self.utci_calculated
        elif metric == TemporospatialMetric.UTCI_INTERPOLATED:
            return self.utci_interpolated
        elif metric == TemporospatialMetric.DBT:
            return self.dry_bulb_temperature
        elif metric == TemporospatialMetric.RH:
            return self.relative_humidity
        elif metric == TemporospatialMetric.WS:
            return self.wind_speed
        elif metric == TemporospatialMetric.WD:
            return self.wind_direction
        elif metric == TemporospatialMetric.EVAP_CLG:
            return self.evaporative_cooling_magnitude
        else:
            raise NotImplementedError()

    @cached_property
    def points(self) -> pd.DataFrame:
        """Obtain the point locations for this SpatialComfort case."""
        return load_points(self.spatial_simulation_directory)

    @property
    def points_xy(self) -> np.ndarray:
        return np.stack([self._points_x, self._points_y], axis=1)

    @cached_property
    def total_rad(self) -> pd.DataFrame:
        """Obtain the Total Irradiance values for this SpatialComfort case."""
        return load_total_irradiance(self.spatial_simulation_directory)

    @cached_property
    def direct_rad(self) -> pd.DataFrame:
        """Obtain the Direct Irradiance values for this SpatialComfort case."""
        return load_direct_irradiance(self.spatial_simulation_directory)

    @cached_property
    def diffuse_rad(self) -> pd.DataFrame:
        """Obtain the Diffuse Irradiance values for this SpatialComfort case."""
        return load_diffuse_irradiance(
            self.spatial_simulation_directory,
            self.total_rad,
            self.direct_rad,
        )

    @cached_property
    def sky_view(self) -> pd.Series:
        """Obtain the Sky View values for this SpatialComfort case."""
        return load_sky_view(self.spatial_simulation_directory)

    @cached_property
    def utci_interpolated(self) -> pd.DataFrame:
        """Obtain the UTCI values (using a rapid interpolation method) for this SpatialComfort
        case.
        """
        return load_universal_thermal_climate_index_interpolated(
            self.spatial_simulation_directory,
            self._unshaded_utci,
            self._shaded_utci,
            self.total_rad,
            self.sky_view,
            self.simulation_result.epw,
        )

    @cached_property
    def utci_calculated(self) -> pd.DataFrame:
        """Obtain the UTCI values (using a a point-wise calculation method) for this SpatialComfort
        case.
        """
        raise NotImplementedError()

    @cached_property
    def dry_bulb_temperature(self) -> pd.DataFrame:
        """Obtain the spatial dry_bulb_temperature values"""
        raise NotImplementedError()

    @cached_property
    def relative_humidity(self) -> pd.DataFrame:
        """Obtain the spatial relative_humidity values"""
        raise NotImplementedError()

    @cached_property
    def wind_speed(self) -> pd.DataFrame:
        """Obtain the spatial wind_speed values"""
        raise NotImplementedError()

    @cached_property
    def wind_direction(self) -> pd.DataFrame:
        """Obtain the spatial wind_direction values"""
        raise NotImplementedError()

    @cached_property
    def evaporative_cooling_magnitude(self) -> pd.DataFrame:
        """Obtain the spatial evaporative_cooling_magnitude values"""
        raise NotImplementedError()

    @cached_property
    def mean_radiant_temperature(self) -> pd.DataFrame:
        """Obtain the MRT values (using a rapid interpolation method) for this SpatialComfort
        case.
        """
        return calculate_mean_radiant_temperature_interpolated(
            self.spatial_simulation_directory,
            self.simulation_result.unshaded_mean_radiant_temperature,
            self.simulation_result.shaded_mean_radiant_temperature,
            self.total_rad,
            self.sky_view,
            self.simulation_result.epw,
        )

    def plot_utci_comfortable_hours(
        self,
        analysis_period: AnalysisPeriod,
        levels: List[float] = None,
        hours: bool = False,
        interpolated: bool = True,
    ) -> Path:
        """Return the path to the comfortable-hours plot."""

        save_path = (
            self._plot_directory
            / f"time_comfortable_{'hours' if hours else 'percentage'}_{describe_analysis_period(analysis_period, True)}.png"
        )

        print(f"- Plotting {save_path.stem}")

        if interpolated:
            z_temp = self.utci_interpolated.iloc[list(analysis_period.hoys_int), :]
        else:
            z_temp = self.utci_calculated.iloc[list(analysis_period.hoys_int), :]
        z = ((z_temp >= 9) & (z_temp <= 26)).sum().values

        if not hours:
            z = z / len(analysis_period.hoys_int) * 100

        fig = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[z],
            levels=np.linspace(10, 90, 101) if levels is None else levels,
            cmap="magma_r",
            extend="both",
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            colorbar_label=f"Hours comfortable (out of a possible {len(analysis_period.hoys_int)})"
            if hours
            else f"% time comfortable (out of {len(analysis_period.hoys_int)} hours)",
            title=f"Time comfortable (9°C-26°C UTCI) for {describe_analysis_period(analysis_period, False)}",
        )

        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", transparent=True)

        return save_path

    def plot_typical_point_in_time(
        self, metric: TemporospatialMetric, month: int, hour: int
    ) -> Path:
        """See title."""

        if not month in range(1, 13, 1):
            raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
        if not hour in range(0, 24, 1):
            raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

        df = self._get_temporospatial_metric(metric)

        if metric == TemporospatialMetric.MRT:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "inferno"
            norm = None
            label = "Mean radiant temperature (°C)"
            extend = "neither"
        elif (metric == TemporospatialMetric.UTCI_CALCULATED) or (
            metric == TemporospatialMetric.UTCI_INTERPOLATED
        ):
            levels = UTCI_LEVELS
            cmap = UTCI_COLORMAP
            norm = UTCI_BOUNDARYNORM
            label = "Universal thermal climate index (°C)"
            extend = "both"
        elif metric == TemporospatialMetric.RAD_TOTAL:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "bone_r"
            norm = None
            label = "Total irradiance (W/m2)"
            extend = "neither"
        elif metric == TemporospatialMetric.RAD_DIRECT:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "bone_r"
            norm = None
            label = "Direct irradiance (W/m2)"
            extend = "neither"
        elif metric == TemporospatialMetric.RAD_DIFFUSE:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "bone_r"
            norm = None
            label = "Diffuse irradiance (W/m2)"
            extend = "neither"
        elif metric == TemporospatialMetric.DBT:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "Reds"
            norm = None
            label = "Dry bulb temperature (°C)"
            extend = "neither"
        elif metric == TemporospatialMetric.RH:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "Blues"
            norm = None
            label = "Relative Humidity (%)"
            extend = "neither"
        elif metric == TemporospatialMetric.WD:
            levels = np.linspace(0, 360, 101)
            cmap = "jet"
            norm = None
            label = "Wind direction (deg from north)"
            extend = "neither"
        elif metric == TemporospatialMetric.WS:
            levels = np.linspace(0, 360, 101)
            cmap = "Spectral"
            norm = None
            label = "Wind speed (m/s)"
            extend = "neither"
        elif metric == TemporospatialMetric.EVAP_CLG:
            levels = np.linspace(df.min().min(), df.max().max(), 101)
            cmap = "YlGnBu"
            norm = None
            label = "Evaporative cooling magnitude (0-1)"
            extend = "neither"
        else:
            raise NotImplementedError()

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

        save_path = (
            self._plot_directory / f"{metric.name.lower()}_{month:02d}_{hour:02d}.png"
        )

        print(f"- Plotting {save_path.stem}")

        fig = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[z],
            levels=levels,
            cmap=cmap,
            extend=extend,
            norm=norm,
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            colorbar_label=label,
            title=f"{calendar.month_abbr[month]} {hour:02d}:00",
        )

        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", transparent=True)

        return save_path

    def plot_sky_view(self) -> Path:
        """Return the path to the sky view plot."""

        save_path = self._plot_directory / "sky_view.png"

        print(f"- Plotting {save_path.stem}")

        levels = np.linspace(0, 100, 101)

        z = self.sky_view.iloc[:, 0].values

        fig = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[z],
            levels=levels,
            cmap="Spectral_r",
            extend="neither",
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            colorbar_label="Proportion of sky visible (%)",
            title="Proportion of sky visible",
        )

        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", transparent=True)

        return save_path

    def summarise_point(
        self,
        point_index: int,
        point_identifier: str,
        metric: TemporospatialMetric = TemporospatialMetric.UTCI_INTERPOLATED,
    ) -> None:

        if metric not in [
            TemporospatialMetric.UTCI_CALCULATED,
            TemporospatialMetric.UTCI_INTERPOLATED,
        ]:
            raise ValueError("This method only applicable for UTCI metrics.")

        # create the collection for the given point index
        point_utci = from_series(
            self._get_temporospatial_metric(metric)
            .iloc[:, point_index]
            .rename("Universal Thermal Climate Index (C)")
        )

        print(f"- Creating pt-location-summary for {point_identifier}")

        # plot point location on total rad plot
        f = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[self.total_rad.mean(axis=0)],
            levels=100,
            cmap=colormap_sequential("grey", "white"),
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            highlight_pts={point_identifier: point_index},
            show_legend_title=False,
        )
        f.savefig(
            self._plot_directory / f"{point_identifier}_location.png",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create openfield UTCI plot
        f = utci_heatmap_histogram(self._unshaded_utci, "Openfield")
        f.savefig(
            self._plot_directory / "Openfield_utci.png",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create openfield UTCI distance to comfortable plot
        f = utci_distance_to_comfortable(self._unshaded_utci, "Openfield")
        f.savefig(
            self._plot_directory / "Openfield_distance_to_comfortable.png",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create point location UTCI plot
        f = utci_heatmap_histogram(point_utci, point_identifier)
        f.savefig(
            self._plot_directory / f"{point_identifier}_utci.png",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create pt location UTCI distance to comfortable plot
        f = utci_distance_to_comfortable(point_utci, point_identifier)
        f.savefig(
            self._plot_directory / f"{point_identifier}_distance_to_comfortable.png",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create pt location UTCI difference
        f = utci_heatmap_difference(
            self._unshaded_utci,
            point_utci,
            f"Difference between Openfield UTCI and {point_identifier} UTCI",
        )
        f.savefig(
            self._plot_directory / f"{point_identifier}_difference.png",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        plt.close("all")

        return None
