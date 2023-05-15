import copy
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from ladybug.wea import EPW, AnalysisPeriod, Wea
from matplotlib import pyplot as plt
from tqdm import tqdm

from .analysis_period import analysis_period_to_boolean


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
    irradiance: Tuple[Tuple[Tuple[float]]] = field(init=False, repr=False, default=None)
    total: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)
    direct: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)
    diffuse: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)
    reflected: Tuple[Tuple[float]] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.irradiance = (
            self._calculate_irradiance() if self.irradiance is None else self.irradiance
        )
        #####
        self.total = tuple([i[0] for i in self.irradiance])
        self.direct = tuple([i[1] for i in self.irradiance])
        self.diffuse = tuple([i[2] for i in self.irradiance])
        self.reflected = tuple([i[3] for i in self.irradiance])

    @property
    def altitudes(self) -> Tuple[float]:
        """The altitudes to calculate."""
        return np.linspace(0, 90, self.n_altitudes)

    @property
    def azimuths(self) -> Tuple[float]:
        """The azimuths to calculate."""
        return np.linspace(0, 360, self.n_azimuths)

    @property
    def combinations(self) -> Tuple[Tuple[float]]:
        """The combinations of azimuths and altitudes to calculate."""
        return np.array(list(itertools.product(self.altitudes, self.azimuths)))

    def total_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod()
    ) -> pd.DataFrame:
        """The total irradiance for each combination of tilt and orientation, within the given analysis period."""
        mask = analysis_period_to_boolean(analysis_period)

        return (
            pd.DataFrame(
                np.array(self.total).T[mask].sum(axis=0),
                index=pd.MultiIndex.from_arrays(
                    self.combinations.T, names=["altitude", "azimuth"]
                ),
            )
            .unstack()
            .droplevel(0, axis=1)
        )

    def diffuse_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod()
    ) -> pd.DataFrame:
        """The diffuse irradiance for each combination of tilt and orientation, within the given analysis period."""
        mask = analysis_period_to_boolean(analysis_period)

        return (
            pd.DataFrame(
                np.array(self.diffuse).T[mask].sum(axis=0),
                index=pd.MultiIndex.from_arrays(
                    self.combinations.T, names=["altitude", "azimuth"]
                ),
            )
            .unstack()
            .droplevel(0, axis=1)
        )

    def direct_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod()
    ) -> pd.DataFrame:
        """The direct irradiance for each combination of tilt and orientation, within the given analysis period."""
        mask = analysis_period_to_boolean(analysis_period)

        return (
            pd.DataFrame(
                np.array(self.direct).T[mask].sum(axis=0),
                index=pd.MultiIndex.from_arrays(
                    self.combinations.T, names=["altitude", "azimuth"]
                ),
            )
            .unstack()
            .droplevel(0, axis=1)
        )

    def reflected_irradiance(
        self, analysis_period: AnalysisPeriod = AnalysisPeriod()
    ) -> pd.DataFrame:
        """The reflected irradiance for each combination of tilt and orientation, within the given analysis period."""
        mask = analysis_period_to_boolean(analysis_period)

        return (
            pd.DataFrame(
                np.array(self.reflected).T[mask].sum(axis=0),
                index=pd.MultiIndex.from_arrays(
                    self.combinations.T, names=["altitude", "azimuth"]
                ),
            )
            .unstack()
            .droplevel(0, axis=1)
        )

    def _calculate_irradiance(self) -> Tuple[Tuple[Tuple[float]]]:
        results = []
        for alt, az in tqdm(self.combinations):
            result = self.wea.directional_irradiance(
                alt, az, self.ground_reflectance, self.isotropic
            )
            results.append([i.values for i in result])

        return results

    def plot_tof(
        self,
        ax: plt.Axes = None,
        rtype: str = "total",
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        **kwargs,
    ):
        """Plot the total, direct, diffuse and reflected irradiance for each combination of tilt and orientation, within the given analysis period."""

        data = getattr(self, f"{rtype}_irradiance")(analysis_period)

        if ax is None:
            ax = plt.gca()

        # TODO - finishe this method!!!!!!!! - using the method commented out below
        raise NotImplementedError("not finished yet!")

    @classmethod
    def from_epw(cls, epw: EPW, n_altitudes: int = 10, n_azimuths: int = 19):
        return cls(
            wea=Wea.from_epw_file(epw.file_path),
            n_altitudes=n_altitudes,
            n_azimuths=n_azimuths,
        )

    @classmethod
    def from_epw_file(cls, epw_file: str, n_altitudes: int = 10, n_azimuths: int = 19):
        return cls(
            wea=Wea.from_epw_file(epw_file),
            n_altitudes=n_altitudes,
            n_azimuths=n_azimuths,
        )


# def radiation_tilt_orientation_factor(
#     radiation_matrix: pd.DataFrame,
#     ax: plt.Axes = None,
#     **kwargs,
# ) -> Figure:
#     """Convert a radiation matrix to a figure showing the radiation tilt and orientation.

#     Args:
#         radiation_matrix (pd.DataFrame):
#             A matrix with altitude index, azimuth columns, and radiation values in Wh/m2.
#         ax (plt.Axes, optional):
#             A matplotlib Axes object. Defaults to None.
#         **kwargs:
#             Additional keyword arguments are passed to the matplotlib plot.

#     Returns:
#         plt.Axes:
#             An Axes object.
#     """

#     if ax is None:
#         ax = plt.gca()

#     # TODO - make this work with an object instead of a dataframe to ensure correct input! And make wqork with kwargs

#     cmap = kwargs.get("cmap", "YlOrRd")
#     title = kwargs.get("title", None)

#     # Construct input values
#     x = np.tile(radiation_matrix.index, [len(radiation_matrix.columns), 1]).T
#     y = np.tile(radiation_matrix.columns, [len(radiation_matrix.index), 1])
#     z = radiation_matrix.values

#     z_max = radiation_matrix.max().max()
#     z_percent = z / z_max * 100

#     # Find location of max value
#     ind = np.unravel_index(np.argmax(z, axis=None), z.shape)

#     # Create figure
#     cf = ax.contourf(y, x, z / 1000, cmap=cmap, levels=10)
#     cl = ax.contour(
#         y,
#         x,
#         z_percent,
#         levels=[50, 60, 70, 80, 90, 95, 99],
#         colors=["w"],
#         linewidths=[1],
#         alpha=0.75,
#         linestyles=[":"],
#     )
#     ax.clabel(cl, fmt="%r %%")

#     ax.scatter(y[ind], x[ind], c="k")
#     ax.text(
#         y[ind] + 2,
#         x[ind] - 2,
#         f"{z_max / 1000:0.0f}kWh/m${{^2}}$/year",
#         ha="left",
#         va="top",
#         c="w",
#     )

#     for spine in ["top", "right"]:
#         ax.spines[spine].set_visible(False)

#     ax.xaxis.set_major_locator(mticker.MultipleLocator(base=30))

#     ax.grid(b=True, which="major", color="white", linestyle=":", alpha=0.25)

#     cb = plt.colorbar(
#         cf,
#         ax=ax,
#         orientation="vertical",
#         drawedges=False,
#         fraction=0.05,
#         aspect=25,
#         pad=0.02,
#         label="kWh/m${^2}$/year",
#     )
#     cb.outline.set_visible(False)
#     cb.add_lines(cl)
#     cb.locator = mticker.MaxNLocator(nbins=10, prune=None)

#     ax.set_xlabel("Panel orientation (clock-wise from North at 0°)")
#     ax.set_ylabel("Panel tilt (0° facing the horizon, 90° facing the sky)")

#     # Title
#     if title is None:
#         ax.set_title("Annual cumulative radiation", x=0, ha="left", va="bottom")
#     else:
#         ax.set_title(
#             f"{title}\nAnnual cumulative radiation", x=0, ha="left", va="bottom"
#         )

#     plt.tight_layout()

#     return ax


# def radiation_tilt_orientation_matrix(
#     epw: EPW, n_altitudes: int = 10, n_azimuths: int = 19
# ) -> pd.DataFrame:
#     """Compute the annual cumulative radiation matrix per surface tilt and orientation, for a
#         given EPW object.
#     Args:
#         epw (EPW):
#             The EPW object for which this calculation is made.
#         n_altitudes (int, optional):
#             The number of altitudes between 0 and 90 to calculate. Default is 10.
#         n_azimuths (int, optional):
#             The number of azimuths between 0 and 360 to calculate. Default is 19.
#     Returns:
#         pd.DataFrame:
#             A table of insolation values for each simulated azimuth and tilt combo.
#     """

#     # TODO - convert this into an object with properties in order to ensure it can be plotted more easily downstream!
#     wea = Wea.from_annual_values(
#         epw.location,
#         epw.direct_normal_radiation.values,
#         epw.diffuse_horizontal_radiation.values,
#         is_leap_year=epw.is_leap_year,
#     )
#     # I do a bit of a hack here, to calculate only the Eastern insolation - then mirror it about
#     # the North-South axis to get the whole matrix
#     altitudes = np.linspace(0, 90, n_altitudes)
#     azimuths = np.linspace(0, 180, n_azimuths)
#     combinations = np.array(list(itertools.product(altitudes, azimuths)))

#     def f(alt_az):
#         return copy.copy(wea).directional_irradiance(alt_az[0], alt_az[1])[0].total

#     with ThreadPoolExecutor() as executor:
#         results = np.array(list(executor.map(f, combinations[0:]))).reshape(
#             len(altitudes), len(azimuths)
#         )
#     temp = pd.DataFrame(results, index=altitudes, columns=azimuths)
#     new_cols = (360 - temp.columns)[::-1][1:]
#     new_vals = temp.values[::-1, ::-1][
#         ::-1, 1:
#     ]  # some weird array transformation stuff here
#     mirrored = pd.DataFrame(new_vals, columns=new_cols, index=temp.index)
#     return pd.concat([temp, mirrored], axis=1)
