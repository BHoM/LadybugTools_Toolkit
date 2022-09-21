from __future__ import annotations

import calendar
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from cached_property import cached_property
from honeybee.model import Model
from ladybug.analysisperiod import AnalysisPeriod
from ladybug_geometry.geometry3d import Point3D
from ladybugtools_toolkit.external_comfort.simulate.simulation_result import \
    SimulationResult
from ladybugtools_toolkit.external_comfort.spatial.load.dbt_epw import dbt_epw
from ladybugtools_toolkit.external_comfort.spatial.load.dbt_evap import \
    dbt_evap
from ladybugtools_toolkit.external_comfort.spatial.load.evap_clg_magnitude import \
    evap_clg_magnitude
from ladybugtools_toolkit.external_comfort.spatial.load.mrt_interpolated import \
    mrt_interpolated
from ladybugtools_toolkit.external_comfort.spatial.load.points import points
from ladybugtools_toolkit.external_comfort.spatial.load.rad_diffuse import \
    rad_diffuse
from ladybugtools_toolkit.external_comfort.spatial.load.rad_direct import \
    rad_direct
from ladybugtools_toolkit.external_comfort.spatial.load.rad_total import \
    rad_total
from ladybugtools_toolkit.external_comfort.spatial.load.rh_epw import rh_epw
from ladybugtools_toolkit.external_comfort.spatial.load.rh_evap import rh_evap
from ladybugtools_toolkit.external_comfort.spatial.load.sky_view import \
    sky_view
from ladybugtools_toolkit.external_comfort.spatial.load.utci_calculated import \
    utci_calculated
from ladybugtools_toolkit.external_comfort.spatial.load.utci_interpolated import \
    utci_interpolated
from ladybugtools_toolkit.external_comfort.spatial.load.wd_epw import wd_epw
from ladybugtools_toolkit.external_comfort.spatial.load.ws_cfd import ws_cfd
from ladybugtools_toolkit.external_comfort.spatial.load.ws_epw import ws_epw
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import \
    SpatialMetric
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_boundarynorm import \
    spatial_metric_boundarynorm
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_colormap import \
    spatial_metric_colormap
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_levels import \
    spatial_metric_levels
from ladybugtools_toolkit.external_comfort.spatial.sky_view_pov import \
    sky_view_pov
from ladybugtools_toolkit.external_comfort.spatial.spatial_comfort_possible import \
    spatial_comfort_possible
from ladybugtools_toolkit.external_comfort.thermal_comfort.utci.utci import \
    utci
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import \
    describe as describe_analysis_period
from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import \
    from_series
from ladybugtools_toolkit.ladybug_extension.location.to_string import \
    to_string as describe_loc
from ladybugtools_toolkit.plot.colormap_sequential import colormap_sequential
from ladybugtools_toolkit.plot.create_triangulation import create_triangulation
from ladybugtools_toolkit.plot.spatial_heatmap import spatial_heatmap
from ladybugtools_toolkit.plot.utci_distance_to_comfortable import \
    utci_distance_to_comfortable
from ladybugtools_toolkit.plot.utci_heatmap_difference import \
    utci_heatmap_difference
from ladybugtools_toolkit.plot.utci_heatmap_histogram import \
    utci_heatmap_histogram
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
        self._points_x = self.points.x.values
        self._points_y = self.points.y.values
        self._triangulation = create_triangulation(
            self._points_x,
            self._points_y,
            alpha=1.1,
            max_iterations=250,
            increment=0.01,
        )

    def _get_metric(self, spatial_metric: SpatialMetric) -> pd.DataFrame:
        """A convenience method to aid in the loading of spatial datasets."""

        if spatial_metric == SpatialMetric.RAD_DIFFUSE:
            return rad_diffuse(
                self.spatial_simulation_directory,
                self.irradiance_total,
                self.irradiance_direct,
            )

        if spatial_metric == SpatialMetric.RAD_DIRECT:
            return rad_direct(self.spatial_simulation_directory)

        if spatial_metric == SpatialMetric.RAD_TOTAL:
            return rad_total(self.spatial_simulation_directory)

        if spatial_metric == SpatialMetric.MRT_INTERPOLATED:
            return mrt_interpolated(
                self.spatial_simulation_directory,
                self.simulation_result.unshaded_mean_radiant_temperature,
                self.simulation_result.shaded_mean_radiant_temperature,
                self.irradiance_total,
                self.sky_view,
                self.simulation_result.epw,
            )

        if spatial_metric == SpatialMetric.DBT_EPW:
            return dbt_epw(
                self.spatial_simulation_directory, self.simulation_result.epw
            )

        if spatial_metric == SpatialMetric.RH_EPW:
            return rh_epw(self.spatial_simulation_directory, self.simulation_result.epw)

        if spatial_metric == SpatialMetric.WD_EPW:
            return wd_epw(self.spatial_simulation_directory, self.simulation_result.epw)

        if spatial_metric == SpatialMetric.WS_EPW:
            return ws_epw(self.spatial_simulation_directory, self.simulation_result.epw)

        if spatial_metric == SpatialMetric.WS_CFD:
            return ws_cfd(self.spatial_simulation_directory, self.simulation_result.epw)

        if spatial_metric == SpatialMetric.EVAP_CLG:
            return evap_clg_magnitude(
                self.spatial_simulation_directory, self.simulation_result.epw
            )

        if spatial_metric == SpatialMetric.DBT_EVAP:
            return dbt_evap(
                self.spatial_simulation_directory, self.simulation_result.epw
            )

        if spatial_metric == SpatialMetric.RH_EVAP:
            return rh_evap(
                self.spatial_simulation_directory, self.simulation_result.epw
            )

        if spatial_metric == SpatialMetric.UTCI_CALCULATED:
            return utci_calculated(
                self.spatial_simulation_directory,
                self.simulation_result.epw,
                self._unshaded_utci,
                self._shaded_utci,
                self.irradiance_total,
                self.sky_view,
            )

        if spatial_metric == SpatialMetric.UTCI_INTERPOLATED:
            return utci_interpolated(
                self.spatial_simulation_directory,
                self._unshaded_utci,
                self._shaded_utci,
                self.irradiance_total,
                self.sky_view,
                self.simulation_result.epw,
            )

        if spatial_metric == SpatialMetric.SKY_VIEW:
            return sky_view(self.spatial_simulation_directory)

        if spatial_metric == SpatialMetric.POINTS:
            return points(self.spatial_simulation_directory)

    def _get_spatial_property(self, spatial_metric: SpatialMetric) -> pd.DataFrame:
        if spatial_metric == SpatialMetric.RAD_DIFFUSE:
            return self.irradiance_diffuse

        if spatial_metric == SpatialMetric.RAD_DIRECT:
            return self.irradiance_direct

        if spatial_metric == SpatialMetric.RAD_TOTAL:
            return self.irradiance_direct

        if spatial_metric == SpatialMetric.MRT_INTERPOLATED:
            return self.mean_radiant_temperature_interpolated

        if spatial_metric == SpatialMetric.DBT_EPW:
            return self.dry_bulb_temperature_epw

        if spatial_metric == SpatialMetric.RH_EPW:
            return self.relative_humidity_epw

        if spatial_metric == SpatialMetric.WD_EPW:
            return self.wind_direction_epw

        if spatial_metric == SpatialMetric.WS_EPW:
            return self.wind_speed_epw

        if spatial_metric == SpatialMetric.WS_CFD:
            return self.wind_speed_cfd

        if spatial_metric == SpatialMetric.EVAP_CLG:
            return self.evaporative_cooling_magnitude

        if spatial_metric == SpatialMetric.DBT_EVAP:
            return self.dry_bulb_temperature_evap

        if spatial_metric == SpatialMetric.RH_EVAP:
            return self.relative_humidity_evap

        if spatial_metric == SpatialMetric.UTCI_CALCULATED:
            return self.universal_thermal_climate_index_calculated

        if spatial_metric == SpatialMetric.UTCI_INTERPOLATED:
            return self.universal_thermal_climate_index_interpolated

        if spatial_metric == SpatialMetric.SKY_VIEW:
            return self.sky_view

        if spatial_metric == SpatialMetric.POINTS:
            return self.points

    @cached_property
    def points(self) -> pd.DataFrame:
        """Obtain the point locations."""
        return self._get_metric(SpatialMetric.POINTS)

    @property
    def points_xy(self) -> np.ndarray:
        """Get the points associated with this object as an array of [[x, y], [x, y], ...]"""
        return np.stack([self._points_x, self._points_y], axis=1)

    @cached_property
    def irradiance_total(self) -> pd.DataFrame:
        """Obtain the Total Irradiance values."""
        return self._get_metric(SpatialMetric.RAD_TOTAL)

    @cached_property
    def irradiance_direct(self) -> pd.DataFrame:
        """Obtain the Direct Irradiance values."""
        return self._get_metric(SpatialMetric.RAD_DIRECT)

    @cached_property
    def irradiance_diffuse(self) -> pd.DataFrame:
        """Obtain the Diffuse Irradiance values."""
        return self._get_metric(SpatialMetric.RAD_DIFFUSE)

    @cached_property
    def sky_view(self) -> pd.DataFrame:
        """Obtain the Sky View values."""
        return self._get_metric(SpatialMetric.SKY_VIEW)

    @cached_property
    def universal_thermal_climate_index_interpolated(self) -> pd.DataFrame:
        """Obtain UTCI values (using a rapid interpolation method)."""
        return self._get_metric(SpatialMetric.UTCI_INTERPOLATED)

    @cached_property
    def universal_thermal_climate_index_calculated(self) -> pd.DataFrame:
        """Obtain UTCI values (using a point-wise calculation method)."""
        return self._get_metric(SpatialMetric.UTCI_CALCULATED)

    @cached_property
    def dry_bulb_temperature_epw(self) -> pd.DataFrame:
        """Obtain the DBT values from the weatherfile."""
        return self._get_metric(SpatialMetric.DBT_EPW)

    @cached_property
    def relative_humidity_epw(self) -> pd.DataFrame:
        """Obtain the RH values from the weatherfile."""
        return self._get_metric(SpatialMetric.RH_EPW)

    @cached_property
    def wind_speed_epw(self) -> pd.DataFrame:
        """Obtain the WS values from the weatherfile."""
        return self._get_metric(SpatialMetric.WS_EPW)

    @cached_property
    def wind_speed_cfd(self) -> pd.DataFrame:
        """Obtain the WS values from a CFD simulation."""
        return self._get_metric(SpatialMetric.WS_CFD)

    @cached_property
    def wind_direction_epw(self) -> pd.DataFrame:
        """Obtain the WD values from the weatherfile."""
        return self._get_metric(SpatialMetric.WD_EPW)

    @cached_property
    def evaporative_cooling_magnitude(self) -> pd.DataFrame:
        """Obtain the spatial evaporative_cooling_magnitude values"""
        return self._get_metric(SpatialMetric.EVAP_CLG)

    @cached_property
    def dry_bulb_temperature_evap(self) -> pd.DataFrame:
        """Obtain the effective DBT values from evaporative cooling."""
        return self._get_metric(SpatialMetric.DBT_EVAP)

    @cached_property
    def relative_humidity_evap(self) -> pd.DataFrame:
        """Obtain the effective RH values from evaporative cooling."""
        return self._get_metric(SpatialMetric.RH_EVAP)

    @cached_property
    def mean_radiant_temperature_interpolated(self) -> pd.DataFrame:
        """Obtain the MRT values (using a rapid interpolation method)."""
        return self._get_metric(SpatialMetric.MRT_INTERPOLATED)

    def plot_comfortable_hours(
        self,
        analysis_period: AnalysisPeriod,
        levels: List[float] = None,
        hours: bool = False,
        metric: SpatialMetric = SpatialMetric.UTCI_INTERPOLATED,
        extension: str = ".png",
    ) -> Path:
        """Return the path to the comfortable-hours plot."""

        if metric not in [
            SpatialMetric.UTCI_INTERPOLATED,
            SpatialMetric.UTCI_CALCULATED,
        ]:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )

        if extension not in [".pdf", ".png"]:
            raise ValueError(
                f"You cannot plot an image with the extension '{extension}'."
            )

        save_path = (
            self._plot_directory
            / f"time_comfortable_{'hours' if hours else 'percentage'}_{describe_analysis_period(analysis_period, True)}{extension}"
        )

        z_temp = self._get_spatial_property(metric).iloc[
            list(analysis_period.hoys_int), :
        ]

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
        self,
        metric: SpatialMetric,
        month: int,
        hour: int,
        levels: List[float] = None,
        cmap: Any = None,
        extension: str = ".png",
        contours: List[float] = None,
    ) -> Path:
        """Create a typical point-in-time plot of the given metric."""

        if metric in [SpatialMetric.POINTS, SpatialMetric.SKY_VIEW]:
            raise ValueError(
                "This type of plot is not possible for the requested metric."
            )

        if extension not in [".pdf", ".png"]:
            raise ValueError(
                f"You cannot plot an image with the extension '{extension}'."
            )

        if not month in range(1, 13, 1):
            raise ValueError(f"Month must be between 1 and 12 inclusive, got {month}")
        if not hour in range(0, 24, 1):
            raise ValueError(f"Hour must be between 0 and 23 inclusive, got {hour}")

        df = self._get_spatial_property(metric)

        if levels is None:
            levels = spatial_metric_levels(metric)
        boundary_norm = spatial_metric_boundarynorm(metric)
        if cmap is None:
            cmap = spatial_metric_colormap(metric)
        if (levels is None) and (boundary_norm is None):
            levels = np.linspace(df.min().min(), df.max().max(), 101)

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
            self._plot_directory
            / f"{metric.name.lower()}_{month:02d}_{hour:02d}{extension}"
        )

        fig = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[z],
            levels=levels,
            cmap=cmap,
            extend="neither",
            norm=boundary_norm,
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            colorbar_label=metric.value,
            title=f"{calendar.month_abbr[month]} {hour:02d}:00",
            contours=contours,
        )

        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", transparent=True)

        return save_path

    def plot_sky_view(self, extension: str = ".png") -> Path:
        """Return the path to the sky view plot."""

        if extension not in [".pdf", ".png"]:
            raise ValueError(
                f"You cannot plot an image with the extension '{extension}'."
            )

        save_path = self._plot_directory / f"sky_view{extension}"

        metric = SpatialMetric.SKY_VIEW
        z = self.sky_view.squeeze().values

        fig = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[z],
            levels=spatial_metric_levels(metric),
            cmap=spatial_metric_colormap(metric),
            extend="neither",
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            colorbar_label=metric.value,
            title="Proportion of sky visible",
        )

        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", transparent=True)

        return save_path

    def plot_spatial_point_locations(self, n: int = 25, extension: str = ".png") -> Path:
        """Return the path to the spatial point locations plot."""
        save_path = self._plot_directory / f"pt_locations{extension}"

        x = self._points_x
        y = self._points_y
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set_aspect("equal")
        ax.scatter(x, y, c="#555555", s=1)
        for i in np.arange(0, len(self._points_x), n):
            ax.scatter(x[i], y[i], c="red", s=2)
            ax.text(
                x[i],
                y[i],
                i,
                ha="center",
                va="center",
                fontsize="small",
            )
        plt.tight_layout()

        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", transparent=True)

        return fig

    def plot_sky_view_pov(
        self,
        point_index: int,
        point_identifier: str,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        show_sunpath: bool = True,
        show_skymatrix: bool = True,
    ) -> Path:
        img = sky_view_pov(
            model=Model.from_hbjson(
                list(self.spatial_simulation_directory.glob("**/*.hbjson"))[0]
            ),
            sensor=Point3D.from_array(
                self.points.iloc[point_index][["x", "y", "z"]].values
            ),
            epw=self.simulation_result.epw,
            analysis_period=AnalysisPeriod(timestep=5),
            cmap=None,
            norm=None,
            data_collection=None,
            density=4,
            show_sunpath=show_sunpath,
            show_skymatrix=show_skymatrix,
            title=f"{describe_loc(self.simulation_result.epw.location)}\n{self.spatial_simulation_directory.stem}\n{describe_analysis_period(analysis_period)}\n{point_identifier}",
        )
        save_path = self._plot_directory / f"{point_identifier}_skyview.png"
        img.save(save_path, dpi=(500, 500))
        return save_path

    def summarise_point(
        self,
        point_index: int,
        point_identifier: str,
        metric: SpatialMetric = SpatialMetric.UTCI_INTERPOLATED,
        extension: str = ".png",
    ) -> None:

        if metric not in [
            SpatialMetric.UTCI_CALCULATED,
            SpatialMetric.UTCI_INTERPOLATED,
        ]:
            raise ValueError("This method only applicable for UTCI metrics.")

        if extension not in [".pdf", ".png"]:
            raise ValueError(
                f"You cannot plot an image with the extension '{extension}'."
            )

        # create the collection for the given point index
        point_utci = from_series(
            self._get_spatial_property(metric)
            .iloc[:, point_index]
            .rename("Universal Thermal Climate Index (C)")
        )

        # plot point location on total rad plot
        f = spatial_heatmap(
            triangulations=[self._triangulation],
            values=[self.irradiance_total.mean(axis=0)],
            levels=100,
            cmap=colormap_sequential("grey", "white"),
            xlims=[self._points_x.min(), self._points_x.max()],
            ylims=[self._points_y.min(), self._points_y.max()],
            highlight_pts={point_identifier: point_index},
            show_legend_title=False,
        )
        f.savefig(
            self._plot_directory / f"{point_identifier}_location{extension}",
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
            self._plot_directory / f"Openfield_distance_to_comfortable{extension}",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create point location UTCI plot
        f = utci_heatmap_histogram(point_utci, point_identifier)
        f.savefig(
            self._plot_directory / f"{point_identifier}_utci{extension}",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create pt location UTCI distance to comfortable plot
        f = utci_distance_to_comfortable(point_utci, point_identifier)
        f.savefig(
            self._plot_directory
            / f"{point_identifier}_distance_to_comfortable{extension}",
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
            self._plot_directory / f"{point_identifier}_difference{extension}",
            dpi=PLOT_DPI,
            transparent=True,
            bbox_inches="tight",
        )

        # create pt location sky view
        self.plot_sky_view_pov(point_index, point_identifier)

        plt.close("all")

        return None
