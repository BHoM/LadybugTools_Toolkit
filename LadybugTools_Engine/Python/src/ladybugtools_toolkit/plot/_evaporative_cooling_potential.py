"""Method for plotting evaporative cooling potential."""

import matplotlib.pyplot as plt
from pathlib import Path
from ladybug.epw import EPW

from python_toolkit.plot.heatmap import heatmap
from ..ladybug_extension.datacollection import collection_to_series
from ..ladybug_extension.location import location_to_string


def evaporative_cooling_potential(epw: EPW, ax: plt.Axes = None, **kwargs) -> plt.Axes:
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

    dpt = collection_to_series(epw.dew_point_temperature)
    dbt = collection_to_series(epw.dry_bulb_temperature)
    ecp = (dbt - dpt).clip(lower=0).rename("Evaporative Cooling Potential (C)")

    if "cmap" not in kwargs:
        kwargs["cmap"] = "GnBu"

    heatmap(series=ecp, ax=ax, **kwargs)
    ax.text(
        1,
        1,
        "*value shown indicate cooling effect from saturating air with moisture (DBT - DPT)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )

    ax.set_title(Path(epw.file_path).name)

    return ax
