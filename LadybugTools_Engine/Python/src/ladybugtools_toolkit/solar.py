"""Methods for handling solar radiation."""

# pylint: disable=E0401
import itertools
import textwrap
import concurrent
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# pylint: enable=E0401

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from ladybug.wea import EPW, AnalysisPeriod, Wea
from matplotlib import pyplot as plt
from tqdm import tqdm

from .bhom import decorator_factory, CONSOLE_LOGGER
from .ladybug_extension.analysisperiod import (
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from .ladybug_extension.location import location_to_string
from .directionbins import DirectionBins
from .ladybug_extension.datacollection import header_to_string

from .plot.utilities import contrasting_color, format_polar_plot


class IrradianceType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()
    REFLECTED = auto()

    def to_string(self) -> str:
        """Get the string representation of the IrradianceType."""
        return self.name.title()


@dataclass(init=True, repr=True, eq=True)
class Solar:
    """An object to handle solar radiation."""

    epw: EPW = field(init=True, repr=True)

    def __post_init__(self):
        if isinstance(self.epw, str | Path):
            self.epw = EPW(self.epw)

    @property
    def wea(self) -> Wea:
        return Wea.from_epw_file(self.epw.file_path)

    @staticmethod
    def altitudes(n_altitudes: int) -> list[float]:
        """Get a list of altitudes."""
        if not isinstance(n_altitudes, int):
            raise ValueError("n_altitudes must be an integer.")
        if n_altitudes < 3:
            raise ValueError("n_altitudes must be an integer >= 3.")
        return np.linspace(0, 90, n_altitudes).tolist()

    @staticmethod
    def azimuths(n_azimuths: int) -> list[float]:
        """Get a list of azimuths."""
        if not isinstance(n_azimuths, int):
            raise ValueError("n_azimuths must be an integer.")
        if n_azimuths < 3:
            raise ValueError("n_azimuths must be an integer >= 3.")
        return np.linspace(0, 360, n_azimuths).tolist()

    @decorator_factory()
    def directional_irradiance_matrix(
        self,
        altitude: float | int = 0,
        n_directions: int = 32,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        reload: bool = True,
    ) -> pd.DataFrame:
        """Calculate the irradiance in W/m2 for each direction, for the given altitude.

        Args:
            altitude (float, optional):
                The altitude of the facade. Defaults to 0 for a vertical surface.
            n_directions (int, optional):
                The number of directions to calculate. Defaults to 32.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            reload (bool, optional):
                reload results if they've already been saved.

        Returns:
            pd.DataFrame:
                A dataframe containing a huge amount of information!
        """
        _wea = self.wea
        db = DirectionBins(directions=n_directions)

        _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_solar"
        _dir.mkdir(exist_ok=True, parents=True)
        sp = (
            _dir
            / f"{Path(self.epw.file_path).stem}_{ground_reflectance}_{isotropic}_{db}_{altitude}.h5"
        )
        if reload and sp.exists():
            CONSOLE_LOGGER.info(f"Loading results from {sp}.")
            return pd.read_hdf(sp, key="df")

        # get headers
        cols = _wea.directional_irradiance(0, 0)
        units = [header_to_string(i.header) for i in cols]
        idx = analysis_period_to_datetimes(cols[0].header.analysis_period)

        results = []
        pbar = tqdm(total=len(db.midpoints), desc="Calculating irradiance")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for _az in db.midpoints:
                futures.append(
                    executor.submit(
                        _wea.directional_irradiance,
                        altitude,
                        _az,
                        ground_reflectance,
                        isotropic,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(n=1)
                results.append(future.result())

        # convert results into a massive array
        headers = []
        for _az in db.midpoints:
            for rad_type, unit in zip(
                *[["Total", "Direct", "Diffuse", "Reflected"], units]
            ):
                headers.append((altitude, _az, rad_type, unit))

        df = pd.DataFrame(
            np.array(results).reshape(len(db.midpoints) * 4, 8760),
            columns=idx,
            index=pd.MultiIndex.from_tuples(
                headers, names=["Altitude", "Azimuth", "Type", "Unit"]
            ),
        ).T

        df.to_hdf(sp, key="df", complevel=9, complib="blosc:zlib")

        return df

    @decorator_factory()
    def detailed_irradiance_matrix(
        self,
        n_altitudes: int = 3,
        n_azimuths: int = 3,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        reload: bool = True,
    ) -> pd.DataFrame:
        """Calculate the irradiance in W/m2 for each combination of tilt and orientation.

        Args:
            n_altitudes (int, optional):
                The number of altitudes to calculate. Defaults to 3.
            n_azimuths (int, optional):
                The number of azimuths to calculate. Defaults to 3.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            reload (bool, optional):
                reload results if they've already been saved.

        Returns:
            pd.DataFrame:
                A dataframe containing a huge amount of information!
        """

        _wea = self.wea

        altitudes = self.altitudes(n_altitudes)
        azimuths = self.azimuths(n_azimuths)
        combinations = list(itertools.product(altitudes, azimuths))

        _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_solar"
        _dir.mkdir(exist_ok=True, parents=True)
        sp = (
            _dir
            / f"{Path(self.epw.file_path).stem}_{ground_reflectance}_{isotropic}_{n_altitudes}_{n_azimuths}.h5"
        )

        if reload and sp.exists():
            CONSOLE_LOGGER.info(f"Loading results from {sp}.")
            return pd.read_hdf(sp, key="df")

        # get headers
        cols = _wea.directional_irradiance(0, 0)
        units = [header_to_string(i.header) for i in cols]
        idx = analysis_period_to_datetimes(cols[0].header.analysis_period)

        results = []
        pbar = tqdm(total=len(combinations), desc="Calculating irradiance")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for _alt, _az in combinations:
                futures.append(
                    executor.submit(
                        _wea.directional_irradiance,
                        _alt,
                        _az,
                        ground_reflectance,
                        isotropic,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(n=1)
                results.append(future.result())

        # convert results into a massive array
        headers = []
        for _alt, _az in combinations:
            for rad_type, unit in zip(
                *[["Total", "Direct", "Diffuse", "Reflected"], units]
            ):
                headers.append((_alt, _az, rad_type, unit))

        df = pd.DataFrame(
            np.array(results).reshape(len(combinations) * 4, 8760),
            columns=idx,
            index=pd.MultiIndex.from_tuples(
                headers, names=["Altitude", "Azimuth", "Type", "Unit"]
            ),
        ).T

        df.to_hdf(sp, key="df", complevel=9, complib="blosc:zlib")

        return df

    @decorator_factory()
    def plot_tilt_orientation_factor(
        self,
        ax: plt.Axes = None,
        n_altitudes: int = 3,
        n_azimuths: int = 3,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        agg: str = "sum",
        **kwargs,
    ) -> plt.Axes:
        """Plot a tilt-orientation factor.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. Defaults to None.
            n_altitudes (int, optional):
                The number of altitudes to calculate. Defaults to 3.
            n_azimuths (int, optional):
                The number of azimuths to calculate. Defaults to 3.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            irradiance_type (IrradianceType, optional):
                The irradiance type to plot. Defaults to IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period. Defaults to AnalysisPeriod().
            agg (str, optional):
                The aggregation method. Defaults to "sum".
            **kwargs:
                Keyword arguments to pass to tricontourf.
        """
        if ax is None:
            ax = plt.gca()

        mtx = (
            self.detailed_irradiance_matrix(
                n_azimuths=n_azimuths,
                n_altitudes=n_altitudes,
                ground_reflectance=ground_reflectance,
                isotropic=isotropic,
            )
            .filter(regex=irradiance_type.to_string())
            .iloc[analysis_period_to_boolean(analysis_period)]
        ).agg(agg, axis=0) / (1000 if agg == "sum" else 1)

        _max = mtx.max()
        _max_alt, _max_az, _, unit = mtx.idxmax()
        if agg == "sum":
            unit = unit.replace("(W/m2)", "kWh/m$^2$").replace("Irradiance ", "")
        else:
            unit = unit.replace("(W/m2)", "Wh/m$^2$").replace("Irradiance ", "")

        if _max == 0:
            raise ValueError(f"No solar radiation within {analysis_period}.")

        tcf = ax.tricontourf(
            mtx.index.get_level_values("Azimuth"),
            mtx.index.get_level_values("Altitude"),
            mtx.values,
            extend="max",
            cmap=kwargs.pop("cmap", "YlOrRd"),
            levels=kwargs.pop("levels", 51),
            **kwargs,
        )

        quantiles = [0.25, 0.5, 0.75, 0.95]
        quantile_values = mtx.quantile(quantiles).values
        tcl = ax.tricontour(
            mtx.index.get_level_values("Azimuth"),
            mtx.index.get_level_values("Altitude"),
            mtx.values,
            levels=quantile_values,
            colors="k",
            linestyles="--",
            alpha=0.5,
        )

        def cl_fmt(x):
            return f"{x:,.0f}{unit}"

        _ = ax.clabel(tcl, fontsize="small", fmt=cl_fmt)
        ax.scatter(_max_az, _max_alt, c="k", s=10, marker="x")
        alt_offset = (90 / 100) * 0.5 if _max_alt <= 45 else -(90 / 100) * 0.5
        az_offset = (360 / 100) * 0.5 if _max_az <= 180 else -(360 / 100) * 0.5
        ha = "left" if _max_az <= 180 else "right"
        va = "bottom" if _max_alt <= 45 else "top"
        ax.text(
            _max_az + az_offset,
            _max_alt + alt_offset,
            f"{_max:,.0f}{unit}\n({_max_az:0.0f}°, {_max_alt:0.0f}°)",
            ha=ha,
            va=va,
            c="k",
            weight="bold",
            size="small",
        )

        # format plot
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=30))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=10))

        cb = plt.colorbar(
            tcf,
            ax=ax,
            orientation="vertical",
            drawedges=False,
            fraction=0.05,
            aspect=25,
            pad=0.02,
            label=unit,
        )
        cb.outline.set_visible(False)
        for quantile_val in quantile_values:
            cb.ax.plot([0, 1], [quantile_val] * 2, "k", ls="--", alpha=0.5)

        ax.set_title(
            (
                f"{location_to_string(self.epw.location)}\n{irradiance_type.to_string()} "
                f"Irradiance ({agg.upper()})\n{describe_analysis_period(analysis_period)}"
            )
        )
        ax.set_xlabel("Panel orientation (clockwise from North at 0°)")
        ax.set_ylabel("Panel tilt (0° facing the horizon, 90° facing the sky)")
        return ax

    @decorator_factory()
    def plot_directional_irradiance(
        self,
        ax: plt.Axes = None,
        altitude: float | int = 0,
        n_directions: int = 32,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        agg: str = "sum",
        labelling: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot the directional irradiance for the given configuration.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. Defaults to None.
            altitude (int, optional):
                The altitude to calculate. Defaults to 0 for vertical surfaces.
            n_directions (int, optional):
                The number of directions to calculate. Defaults to 32.
            ground_reflectance (float, optional):
                The ground reflectance. Defaults to 0.2.
            isotropic (bool, optional):
                Calculate isotropic diffuse irradiance. Defaults to True.
            irradiance_type (IrradianceType, optional):
                The irradiance type to plot. Defaults to IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period. Defaults to AnalysisPeriod().
            agg (str, optional):
                The aggregation method. Defaults to "sum".
            labelling (bool, optional):
                Label the plot. Defaults to True.
            **kwargs:
                Keyword arguments to pass to tricontourf.

        Returns:
            plt.Axes: The matplotlib axes.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        format_polar_plot(ax)

        cmap = plt.get_cmap(
            kwargs.pop(
                "cmap",
                "Spectral_r",
            )
        )
        width = kwargs.pop("width", 0.9)
        vmin = kwargs.pop("vmin", 0)

        # plot data
        data = self.directional_irradiance_matrix(
            altitude=altitude,
            n_directions=n_directions,
            ground_reflectance=ground_reflectance,
            isotropic=isotropic,
        ).filter(regex=irradiance_type.to_string()).iloc[
            analysis_period_to_boolean(analysis_period)
        ].agg(
            agg, axis=0
        ) / (
            1000 if agg == "sum" else 1
        )

        _max = data.max()
        _, _max_az, _, unit = data.idxmax()
        if agg == "sum":
            unit = unit.replace("(W/m2)", "kWh/m$^2$").replace("Irradiance ", "")
        else:
            unit = unit.replace("(W/m2)", "Wh/m$^2$").replace("Irradiance ", "")
        if _max == 0:
            raise ValueError(f"No solar radiation within {analysis_period}.")

        vmax = kwargs.pop("vmax", data.max())
        colors = [cmap(i) for i in np.interp(data.values, [vmin, vmax], [0, 1])]
        thetas = np.deg2rad(data.index.get_level_values("Azimuth"))
        radiis = data.values
        bars = ax.bar(
            thetas,
            radiis,
            zorder=2,
            width=width * np.deg2rad(360 / len(thetas)),
            color=colors,
        )

        # colorbar
        if (data.min() < vmin) and (data.max() > vmax):
            extend = "both"
        elif data.min() < vmin:
            extend = "min"
        elif data.max() > vmax:
            extend = "max"
        else:
            extend = "neither"
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, orientation="vertical", label=unit, extend=extend)
        cb.outline.set_visible(False)

        # labelling
        if labelling:
            for rect, idx, val, colr in list(
                zip(
                    *[bars, data.index.get_level_values("Azimuth"), data.values, colors]
                )
            ):
                if len(set(data.values)) == 1:
                    ax.text(
                        0,
                        0,
                        textwrap.fill(
                            f"{val:,.0f}{unit} {agg} total insolation",
                            16,
                        ),
                        ha="center",
                        va="center",
                        bbox={
                            "ec": "none",
                            "fc": "w",
                            "alpha": 0.5,
                            "boxstyle": "round,pad=0.3",
                        },
                    )
                    break
                if val == data.max():
                    rect.set_edgecolor("k")
                if val > data.max() / 1.5:
                    ax.text(
                        np.deg2rad(idx),
                        val,
                        f" {val:,.0f}{unit} ",
                        rotation_mode="anchor",
                        rotation=(-idx + 90) if idx < 180 else 180 + (-idx + 90),
                        ha="right" if idx < 180 else "left",
                        va="center",
                        fontsize="xx-small",
                        c=contrasting_color(colr)
                        # bbox=dict(ec="none", fc="w", alpha=0.5, boxstyle="round,pad=0.3"),
                    )

        _ = ax.set_title(
            (
                f"{location_to_string(self.epw.location)}\n"
                f"{irradiance_type.to_string()} irradiance ({agg.upper()}) at {altitude}° tilt\n"
                f"{describe_analysis_period(analysis_period)}"
            )
        )

        return ax
