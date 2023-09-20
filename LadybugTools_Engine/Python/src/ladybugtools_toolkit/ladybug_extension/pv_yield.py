from __future__ import annotations

import copy
import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.wea import EPW, AnalysisPeriod, Wea
from matplotlib import pyplot as plt
from tqdm import tqdm

from .analysis_period import (
    analysis_period_to_boolean,
    analysis_period_to_datetimes,
    describe_analysis_period,
)
from .location import location_to_string


class IrradianceType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()
    REFLECTED = auto()

    def to_string(self) -> str:
        """Get the string representation of the IrradianceType."""
        return self.name.title()


class IrradianceUnit(Enum):
    """Irradiance units."""

    WH_M2 = auto()
    KWH_M2 = auto()
    MWH_M2 = auto()

    def to_string(self) -> str:
        """Get the string representation of the IrradianceUnit."""
        d = {
            self.WH_M2.value: "Wh/m$^{2}$",
            self.KWH_M2.value: "kWh/m$^{2}$",
            self.MWH_M2.value: "MWh/m$^{2}$",
        }
        return d[self.value]

    @property
    def multiplier(self) -> float:
        """The multiplier for the unit."""
        d = {
            self.WH_M2.value: 1,
            self.KWH_M2.value: 0.001,
            self.MWH_M2.value: 0.000001,
        }
        return d[self.value]


@dataclass(init=True, repr=True, eq=True)
class PVYieldMatrix:
    """Compute the annual cumulative radiation matrix per surface tilt and orientation, for a given Wea object.

    Args:
        wea (Wea):
            The Wea object for which this calculation is made.
        n_altitudes (int, optional):
            The number of altitudes between 0 and 90 to calculate. Default is 10.
        n_azimuths (int, optional):
            The number of azimuths between 0 and 360 to calculate. Default is 19.
        ground_reflectance (float, optional):
            The ground reflectance to use for the calculation. Default is 0.2.
        isotropic (bool, optional):
            An isotropic sky assumes an even distribution of diffuse
            irradiance across the sky while an anisotropic sky places more
            diffuse irradiance near the solar disc.
            Set to True to use isotropic sky model. Default is False.

    Returns:
        pd.DataFrame:
            A table of insolation values for each simulated azimuth and tilt combo.
    """

    wea: Wea = field(init=True, repr=True)
    n_altitudes: int = field(init=True, repr=True, default=10)  # 10
    n_azimuths: int = field(init=True, repr=True, default=19)  # 19
    ground_reflectance: float = field(init=True, repr=True, default=0.2)
    isotropic: bool = field(init=True, repr=True, default=False)
    ###########
    _calculated_irradiance: Tuple[Tuple[Tuple[float]]] = field(
        init=False, repr=False, default=None
    )
    _total: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)
    _direct: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)
    _diffuse: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)
    _reflected: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.n_altitudes <= 2:
            raise ValueError("n_altitudes must be greater than 2.")
        if self.n_azimuths <= 2:
            raise ValueError("n_azimuths must be greater than 2.")

        self._calculated_irradiance = (
            self._calculate_irradiance()
            if self._calculated_irradiance is None
            else self._calculated_irradiance
        )
        #####
        self._total = tuple([i[0] for i in self._calculated_irradiance])
        self._direct = tuple([i[1] for i in self._calculated_irradiance])
        self._diffuse = tuple([i[2] for i in self._calculated_irradiance])
        self._reflected = tuple([i[3] for i in self._calculated_irradiance])

    @property
    def altitudes(self) -> Tuple[float]:
        """The altitudes to calculate."""
        return np.linspace(0, 90, self.n_altitudes)

    @property
    def azimuths(self) -> Tuple[float]:
        """The azimuths to calculate."""
        return np.linspace(0, 360, self.n_azimuths)

    @property
    def _alt_az_combinations(self) -> Tuple[Tuple[float]]:
        """The combinations of azimuths and altitudes to calculate."""
        return np.array(list(itertools.product(self.altitudes, self.azimuths)))

    def get_irradiance(
        self,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        agg: str = "sum",
    ) -> pd.DataFrame:
        """Get the irradiance for the given irradiance type.

        Args:
            irradiance_type (IrradianceType, optional):
                The irradiance type to get. Default is IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period for which to calculate the irradiance. Default is AnalysisPeriod().
            agg (str, optional):
                The aggregation method to use for the calculation. Default is "sum".

        Returns:
            pd.DataFrame: A table of irradiance values for each simulated azimuth and tilt combo.
        """
        if irradiance_type.value == IrradianceType.TOTAL.value:
            return self.total_irradiance(analysis_period=analysis_period, agg=agg)
        if irradiance_type.value == IrradianceType.DIRECT.value:
            return self.direct_irradiance(analysis_period=analysis_period, agg=agg)
        if irradiance_type.value == IrradianceType.DIFFUSE.value:
            return self.diffuse_irradiance(analysis_period=analysis_period, agg=agg)
        if irradiance_type.value == IrradianceType.REFLECTED.value:
            return self.reflected_irradiance(analysis_period=analysis_period, agg=agg)

        raise ValueError("irradiance_type must be a IrradianceType.")

    def total_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod(), agg: str = "sum"
    ) -> pd.DataFrame:
        """The total irradiance for each combination of tilt and orientation, within the given analysis period.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period for which to calculate the total irradiance. Default is AnalysisPeriod().
            agg (str, optional):
                The aggregation method to use for the calculation. Default is "sum".
        Returns:
            pd.DataFrame:
                A table of total irradiance values for each simulated azimuth and tilt combo.
        """
        mask = analysis_period_to_boolean(analysis_period)
        dts = analysis_period_to_datetimes(analysis_period)

        temp = pd.DataFrame(
            np.array(self._total).T[mask].T,
            columns=dts,
            index=pd.MultiIndex.from_arrays(
                self._alt_az_combinations.T, names=["altitude", "azimuth"]
            ),
        )
        temp = temp.agg(agg, axis=1)
        temp = temp.unstack()

        return temp

    def diffuse_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod(), agg: str = "sum"
    ) -> pd.DataFrame:
        """The diffuse irradiance for each combination of tilt and orientation, within the given analysis period.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period for which to calculate the total irradiance. Default is AnalysisPeriod().
            agg (str, optional):
                The aggregation method to use for the calculation. Default is "sum".
        Returns:
            pd.DataFrame:
                A table of diffuse irradiance values for each simulated azimuth and tilt combo.
        """
        mask = analysis_period_to_boolean(analysis_period)
        dts = analysis_period_to_datetimes(analysis_period)

        temp = pd.DataFrame(
            np.array(self._diffuse).T[mask].T,
            columns=dts,
            index=pd.MultiIndex.from_arrays(
                self._alt_az_combinations.T, names=["altitude", "azimuth"]
            ),
        )
        temp = temp.agg(agg, axis=1)
        temp = temp.unstack()

        return temp

    def direct_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod(), agg: str = "sum"
    ) -> pd.DataFrame:
        """The direct irradiance for each combination of tilt and orientation, within the given analysis period.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period for which to calculate the total irradiance. Default is AnalysisPeriod().
            agg (str, optional):
                The aggregation method to use for the calculation. Default is "sum".
        Returns:
            pd.DataFrame:
                A table of direct irradiance values for each simulated azimuth and tilt combo.
        """
        mask = analysis_period_to_boolean(analysis_period)
        dts = analysis_period_to_datetimes(analysis_period)

        temp = pd.DataFrame(
            np.array(self._direct).T[mask].T,
            columns=dts,
            index=pd.MultiIndex.from_arrays(
                self._alt_az_combinations.T, names=["altitude", "azimuth"]
            ),
        )
        temp = temp.agg(agg, axis=1)
        temp = temp.unstack()

        return temp

    def reflected_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod(), agg: str = "sum"
    ) -> pd.DataFrame:
        """The reflected irradiance for each combination of tilt and orientation, within the given analysis period.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period for which to calculate the total irradiance. Default is AnalysisPeriod().
            agg (str, optional):
                The aggregation method to use for the calculation. Default is "sum".
        Returns:
            pd.DataFrame:
                A table of reflected irradiance values for each simulated azimuth and tilt combo.
        """
        mask = analysis_period_to_boolean(analysis_period)
        dts = analysis_period_to_datetimes(analysis_period)

        temp = pd.DataFrame(
            np.array(self._reflected).T[mask].T,
            columns=dts,
            index=pd.MultiIndex.from_arrays(
                self._alt_az_combinations.T, names=["altitude", "azimuth"]
            ),
        )
        temp = temp.agg(agg, axis=1)
        temp = temp.unstack()

        return temp

    def _calculate_irradiance(self) -> Tuple[Tuple[Tuple[float]]]:
        """Calculate the irradiance in W/m2 for each combination of tilt and orientation."""
        results = []
        pbar = tqdm(self._alt_az_combinations)
        for _alt, _az in pbar:
            pbar.set_description(
                f"Calculating irradiance for (Altitude: {_alt:0.1f}°, Azimuth: {_az:0.1f}°)"
            )
            result = self.wea.directional_irradiance(
                _alt, _az, self.ground_reflectance, self.isotropic
            )
            results.append([i.values for i in result])

        return results

    def _calculate_irradiance_threaded(self) -> Tuple[Tuple[Tuple[float]]]:
        def task(alt_az):
            _alt, _az = alt_az
            result = self.wea.directional_irradiance(
                _alt, _az, self.ground_reflectance, self.isotropic
            )
            return [i.values for i in result]

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(task, i) for i in self._alt_az_combinations]
            for future in futures:
                # retrieve the result
                results.append(future.result())
        return results

    def plot_tof(
        self,
        ax: plt.Axes = None,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        agg: str = "sum",
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        irradiance_unit: IrradianceUnit = IrradianceUnit.WH_M2,
        **kwargs,
    ) -> plt.Axes:
        """Plot the total, direct, diffuse and reflected irradiance for each combination of tilt and orientation, within the given analysis period."""

        if ax is None:
            ax = plt.gca()

        default_kwargs = {"cmap": "YlOrRd", "levels": 100}
        kwargs = {**default_kwargs, **kwargs}

        # prepare data for plotting
        data = (
            (
                self.get_irradiance(
                    irradiance_type=irradiance_type,
                    agg=agg,
                    analysis_period=analysis_period,
                )
                * irradiance_unit.multiplier
            )
            .unstack()
            .reset_index()
            .rename(columns={0: "vals"})
        )
        max_idx = data.sort_values("vals", ascending=False).index[0]
        max_alt = data.altitude[max_idx]
        max_az = data.azimuth[max_idx]
        max_val = data.vals[max_idx]
        if max_val == 0:
            warnings.warn("No irradiance values found for the given analysis period.")

        # set defaults for plot kwargs
        title = kwargs.pop(
            "title",
            f"{location_to_string(self.wea.location)}\n{describe_analysis_period(analysis_period)} ({irradiance_type.to_string()}, {agg})",
        )
        quantiles = kwargs.pop("quantiles", [0.25, 0.5, 0.75, 0.95])

        # populate plot
        tcf = ax.tricontourf(
            data.azimuth.values,
            data.altitude.values,
            data.vals.values,
            extend="max",
            **kwargs,
        )
        if max_val != 0:
            quantile_vals = [max_val * i for i in quantiles]
            tcl = ax.tricontour(
                data.azimuth.values,
                data.altitude.values,
                data.vals.values / max_val,
                levels=quantiles,
                colors="k",
                linestyles="--",
                alpha=0.5,
            )

            def cl_fmt(x):
                return f"{x:.0%} ({max_val * x:,.0f})"

            _ = ax.clabel(tcl, fmt=cl_fmt)

            # maximium pt
            ax.scatter(max_az, max_alt, c="k", s=10, marker="x")
            alt_offset = (90 / 100) * 0.5 if max_alt <= 45 else -(90 / 100) * 0.5
            az_offset = (360 / 100) * 0.5 if max_az <= 180 else -(360 / 100) * 0.5
            ha = "left" if max_az <= 180 else "right"
            va = "bottom" if max_alt <= 45 else "top"
            ax.text(
                max_az + az_offset,
                max_alt + alt_offset,
                f"{max_val:,.0f}{irradiance_unit.to_string()}\n({max_az:0.0f}°, {max_alt:0.0f}°)",
                ha=ha,
                va=va,
                c="k",
                weight="bold",
                size="small",
            )
            ax.axvline(max_az, 0, max_alt / 90, c="w", ls=":", lw=0.5, alpha=0.25)
            ax.axhline(max_alt, 0, max_az / 360, c="w", ls=":", lw=0.5, alpha=0.25)

        # format plot
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=30))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=10))

        def cb_fmt(x, pos):
            return f"{x:,.0f}"

        cb = plt.colorbar(
            tcf,
            ax=ax,
            orientation="vertical",
            drawedges=False,
            fraction=0.05,
            aspect=25,
            pad=0.02,
            label=irradiance_unit.to_string(),
            format=mticker.FuncFormatter(cb_fmt),
        )
        cb.outline.set_visible(False)
        cb.locator = mticker.MaxNLocator(nbins=10, prune=None)
        if max_val != 0:
            for quantile_val in quantile_vals:
                cb.ax.plot([0, 1], [quantile_val] * 2, "k", ls="--", alpha=0.5)

        # add decorators
        ax.set_title(title)
        ax.set_xlabel("Panel orientation (clockwise from North at 0°)")
        ax.set_ylabel("Panel tilt (0° facing the horizon, 90° facing the sky)")
        plt.tight_layout()

        return ax

    @classmethod
    def from_epw(
        cls, epw: EPW, n_altitudes: int = 10, n_azimuths: int = 19
    ) -> PVYieldMatrix:
        """Create this object from an EPW file.

        Args:
            epw (EPW):
                The EPW file to use (as a ladybug object).
            n_altitudes (int, optional):
                The number of altitudes to use. Defaults to 10.
            n_azimuths (int, optional):
                The number of azimuths to use. Defaults to 19.

        Returns:
            PVYieldMatrix:
                The PVYieldMatrix object.
        """
        return cls(
            wea=Wea.from_epw_file(epw.file_path),
            n_altitudes=n_altitudes,
            n_azimuths=n_azimuths,
        )

    @classmethod
    def from_epw_file(cls, epw_file: str, n_altitudes: int = 10, n_azimuths: int = 19):
        """Create this object from an EPW file.

        Args:
            epw_file (str):
                The EPW file to use (as a path).
            n_altitudes (int, optional):
                The number of altitudes to use. Defaults to 10.
            n_azimuths (int, optional):
                The number of azimuths to use. Defaults to 19.

        Returns:
            PVYieldMatrix:
                The PVYieldMatrix object.
        """
        return cls(
            wea=Wea.from_epw_file(epw_file),
            n_altitudes=n_altitudes,
            n_azimuths=n_azimuths,
        )
