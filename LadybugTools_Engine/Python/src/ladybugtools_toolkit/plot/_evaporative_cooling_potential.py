"""Method for plotting evaporative cooling potential."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ladybug.epw import EPW

from python_toolkit.plot.heatmap import heatmap

from ..ladybug_extension.datacollection import collection_to_series


def evaporative_cooling_potential(
    dbt: pd.Series, dpt: pd.Series, ax: plt.Axes = None, agg_year: bool = False, agg: str = "mean", **kwargs
) -> plt.Axes:
    """Plot evaporative cooling potential (DBT - DPT).

    Args:
        dbt (pd.Series):
            A pandas Series containing Dry Bulb Temperature data.
        dpt (pd.Series):
            A pandas Series containing Dew Point Temperature data.
        ax (plt.Axes, optional):
            The matplotlib axes to plot the figure on. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        plt.Axes: The matplotlib axes.
    """

    if len(dbt) != len(dpt):
        raise ValueError("The length of the two series must be the same.")

    if any(dbt.index != dpt.index):
        raise ValueError("The indices of the two series must be the same.")

    if ax is None:
        ax = plt.gca()

    ecp = (dbt - dpt).clip(lower=0).rename("Evaporative Cooling Potential (C)")

    if agg_year:
        # check for presence of Feb 29 in ecp index
        if len(ecp[(ecp.index.month == 2) & (ecp.index.day == 29)]) != 0:
            idx = pd.date_range(start="2016-01-01", periods=8784, freq="h")
        else:
            idx = pd.date_range(start="2017-01-01", periods=8760, freq="h")

        ecp = ecp.groupby([ecp.index.month, ecp.index.day, ecp.index.hour]).agg(agg)
        ecp.index = idx

    if "cmap" not in kwargs:
        kwargs["cmap"] = "GnBu"

    heatmap(series=ecp, ax=ax, **kwargs)
    ax.text(
        1,
        1,
        "*values shown indicate cooling effect from saturating air with moisture (DBT - DPT)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )

    return ax


def evaporative_cooling_potential_epw(epw: EPW, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """Plot evaporative cooling potential (DBT - DPT).

    Args:
        epw (EPW):
            An EPW object.
        ax (plt.Axes, optional):
            The matplotlib axes to plot the figure on. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        plt.Axes: The matplotlib axes.
    """

    if ax is None:
        ax = plt.gca()

    evaporative_cooling_potential(
        dbt=collection_to_series(epw.dry_bulb_temperature), dpt=collection_to_series(epw.dew_point_temperature), ax=ax, **kwargs
    )

    ax.set_title(Path(epw.file_path).name)

    return ax
