from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ...plot.colormaps import UTCI_BOUNDARYNORM, UTCI_COLORMAP, UTCI_LEVELS


class SpatialMetric(Enum):
    """A set of metrics that can be represented spatially."""

    DBT_EPW = auto()
    DBT_EVAP = auto()
    EVAP_CLG = auto()
    MRT_INTERPOLATED = auto()
    POINTS = auto()  # single value - not tabular
    RAD_DIFFUSE = auto()
    RAD_DIRECT = auto()
    RAD_TOTAL = auto()
    RH_EPW = auto()
    RH_EVAP = auto()
    SKY_VIEW = auto()  # single value - not tabular
    UTCI_CALCULATED = auto()
    UTCI_INTERPOLATED = auto()
    WD_EPW = auto()
    WS_CFD = auto()
    WS_EPW = auto()
    DIRECT_SUN_HOURS = auto()  # single value - not tabular

    def filepath(self, simulation_directory: Path) -> Path:
        """Return the expected filepath for a given SpatialMetric.

        Args:
            simulation_directory (Path):
                A location containing simulated spatial thermal comfort results.

        Returns:
            Path:
                The path expected for any stored metric dataset.
        """
        return simulation_directory / f"{self.name.lower()}.parquet"

    def tricontourf_kwargs(self) -> Dict[str, Any]:
        """kwargs for tricontourf plot formatting of the given SpatialMetric."""
        cases = {
            SpatialMetric.DBT_EPW.value: {
                "cmap": plt.get_cmap("YlOrRd"),
            },
            SpatialMetric.MRT_INTERPOLATED.value: {
                "cmap": plt.get_cmap("inferno"),
            },
            SpatialMetric.RAD_DIFFUSE.value: {
                "cmap": plt.get_cmap("bone_r"),
            },
            SpatialMetric.RAD_DIRECT.value: {
                "cmap": plt.get_cmap("bone_r"),
            },
            SpatialMetric.RAD_TOTAL.value: {
                "cmap": plt.get_cmap("bone_r"),
            },
            SpatialMetric.RH_EPW.value: {
                "levels": np.linspace(0, 100, 101),
                "cmap": plt.get_cmap("YlGnBu"),
            },
            SpatialMetric.SKY_VIEW.value: {
                "levels": np.linspace(0, 100, 11),
                "cmap": plt.get_cmap("Spectral_r"),
            },
            SpatialMetric.UTCI_CALCULATED.value: {
                "levels": UTCI_LEVELS,
                "cmap": UTCI_COLORMAP,
                "norm": UTCI_BOUNDARYNORM,
            },
            SpatialMetric.UTCI_INTERPOLATED.value: {
                "levels": UTCI_LEVELS,
                "cmap": UTCI_COLORMAP,
                "norm": UTCI_BOUNDARYNORM,
            },
            SpatialMetric.WD_EPW.value: {
                "levels": np.linspace(0, 360, 17),
                "cmap": plt.get_cmap("twilight"),
            },
            SpatialMetric.WS_CFD.value: {
                "cmap": plt.get_cmap("YlGnBu"),
                "levels": np.linspace(0, 12, 13),
                "extend": "max",
            },
            SpatialMetric.WS_EPW.value: {
                "cmap": plt.get_cmap("YlGnBu"),
                "levels": np.linspace(0, 12, 13),
                "extend": "max",
            },
            SpatialMetric.DIRECT_SUN_HOURS.value: {
                "levels": np.linspace(0, 10, 11),
                "cmap": plt.get_cmap("rainbow"),
                "extend": "max",
            },
        }
        try:
            return cases[self.value]
        except KeyError as exc:
            raise KeyError(f"tricontourf_kwargs not defined for {self}.") from exc

    def tricontour_kwargs(self) -> Dict[str, Any]:
        """kwargs for tricontour plot formatting of the given SpatialMetric."""
        cases = {
            SpatialMetric.DBT_EPW.value: {
                "levels": [],
            },
            SpatialMetric.MRT_INTERPOLATED.value: {
                "levels": [],
            },
            SpatialMetric.RAD_DIFFUSE.value: {
                "levels": [],
            },
            SpatialMetric.RAD_DIRECT.value: {
                "levels": [],
            },
            SpatialMetric.RAD_TOTAL.value: {
                "levels": [],
            },
            SpatialMetric.RH_EPW.value: {
                "levels": [],
            },
            SpatialMetric.SKY_VIEW.value: {
                "levels": [25, 50, 75],
                "colors": "k",
                "linestyles": ["-", "-"],
                "linewidths": [0.5, 0.5],
                "alpha": 0.5,
            },
            SpatialMetric.UTCI_CALCULATED.value: {
                "levels": [],
            },
            SpatialMetric.UTCI_INTERPOLATED.value: {
                "levels": [],
            },
            SpatialMetric.WD_EPW.value: {
                "levels": [],
            },
            SpatialMetric.WS_CFD.value: {
                "levels": [],
            },
            SpatialMetric.WS_EPW.value: {
                "levels": [],
            },
            SpatialMetric.DIRECT_SUN_HOURS.value: {
                "levels": [],
                # "colors": ["r", "g"],
                # "linestyles": ["-", "-"],
                # "linewidths": [1, 1],
            },
        }
        try:
            return cases[self.value]
        except KeyError as exc:
            raise KeyError(f"tricontourf_kwargs not defined for {self}.") from exc

    def description(self) -> str:
        """Return the human readable description of this metric."""
        cases = {
            SpatialMetric.RAD_DIFFUSE.value: "Diffuse Radiation (W/m²)",
            SpatialMetric.RAD_DIRECT.value: "Direct Radiation (W/m²)",
            SpatialMetric.RAD_TOTAL.value: "Total Radiation (W/m²)",
            SpatialMetric.DBT_EPW.value: "Dry-Bulb Temperature (°C) - from EPW",
            SpatialMetric.RH_EPW.value: "Relative Humidity (%) - from EPW",
            SpatialMetric.WD_EPW.value: "Wind Direction (deg) - from EPW",
            SpatialMetric.WS_EPW.value: "Wind Speed (m/s) - from EPW",
            SpatialMetric.WS_CFD.value: "Wind Speed (m/s) - from CFD",
            SpatialMetric.EVAP_CLG.value: "Evaporative Cooling Magnitude (0-1)",
            SpatialMetric.DBT_EVAP.value: "Dry-Bulb Temperature (°C) - inc. moisture effects",
            SpatialMetric.RH_EVAP.value: "Relative Humidity (%) - inc. moisture effects",
            SpatialMetric.MRT_INTERPOLATED.value: "Mean Radiant Temperature (°C) - interpolated",
            SpatialMetric.UTCI_CALCULATED.value: "Universal Thermal Climate Index (°C) - calculated",
            SpatialMetric.UTCI_INTERPOLATED.value: "Universal Thermal Climate Index (°C) - interpolated",
            SpatialMetric.SKY_VIEW.value: "Sky View (%)",
            SpatialMetric.POINTS.value: "Points (x, y, z)",
            SpatialMetric.DIRECT_SUN_HOURS.value: "Direct Sun Hours (hours)",
        }

        try:
            return cases[self.value]
        except KeyError as exc:
            raise KeyError(f"A description is not defined for {self}.") from exc

    def is_temporal(self) -> bool:
        """Return False if the metric chosen is not temporal."""
        cases = {
            SpatialMetric.DBT_EPW.value: True,
            SpatialMetric.DBT_EVAP.value: True,
            SpatialMetric.EVAP_CLG.value: True,
            SpatialMetric.MRT_INTERPOLATED.value: True,
            SpatialMetric.POINTS.value: False,
            SpatialMetric.RAD_DIFFUSE.value: True,
            SpatialMetric.RAD_DIRECT.value: True,
            SpatialMetric.RAD_TOTAL.value: True,
            SpatialMetric.RH_EPW.value: True,
            SpatialMetric.RH_EVAP.value: True,
            SpatialMetric.SKY_VIEW.value: False,
            SpatialMetric.UTCI_CALCULATED.value: True,
            SpatialMetric.UTCI_INTERPOLATED.value: True,
            SpatialMetric.WD_EPW.value: True,
            SpatialMetric.WS_CFD.value: True,
            SpatialMetric.WS_EPW.value: True,
            SpatialMetric.DIRECT_SUN_HOURS.value: False,
        }
        try:
            return cases[self.value]
        except KeyError as exc:
            raise KeyError(f"{self} can not be used!") from exc
