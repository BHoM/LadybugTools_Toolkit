import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

# TODO - method to plot UTCI 2d-histogram

import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ladybug_extension.datacollection import to_series, from_series
from external_comfort.plot.colours import (
    UTCICategory,
    UTCI_COLOURMAP,
    UTCI_COLOURMAP_NORM,
    UTCI_BOUNDS,
    UTCI_CATEGORIES,
    UTCI_COLOURS,
)

WIDTH = 15
HEIGHT = 5

pd.plotting.register_matplotlib_converters()


def utci_heatmap(
    collection: HourlyContinuousCollection,
    invert_y: bool = False,
    title: str = None,
) -> Figure:
    """Create a day/time heatmap for a single years UTCI.

    Args:
        collection (HourlyContinuousCollection): A ladybug HourlyContinuousCollection containing annual hourly UTCI values.
        invert_y (bool, optional): Invert the y-axis to make the start of the day the top of the y-axis. Defaults to False.
        title (str, optional): The string to set as the title.

    Returns:
        Figure: A matplotlib figure object.
    """

    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )

    # Instantiate figure
    fig = plt.figure(figsize=(WIDTH, HEIGHT), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[4, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.75)

    # Construct series
    series = to_series(collection)

    # Add heatmap
    heatmap = heatmap_ax.imshow(
        pd.pivot_table(
            series.to_frame(),
            index=series.index.time,
            columns=series.index.date,
            values=series.name,
        ).values[::-1],
        norm=UTCI_COLOURMAP_NORM,
        extent=[
            mdates.date2num(series.index.min()),
            mdates.date2num(series.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        cmap=UTCI_COLOURMAP,
        interpolation="none",
    )
    heatmap_ax.xaxis_date()
    heatmap_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    heatmap_ax.yaxis_date()
    heatmap_ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if invert_y:
        heatmap_ax.invert_yaxis()
    heatmap_ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(heatmap_ax.get_xticklabels(), ha="left", color="k")
    plt.setp(heatmap_ax.get_yticklabels(), color="k")
    for spine in ["top", "bottom", "left", "right"]:
        heatmap_ax.spines[spine].set_visible(False)
        heatmap_ax.spines[spine].set_color("k")
    heatmap_ax.grid(b=True, which="major", color="k", linestyle=":", alpha=0.5)

    # Add colorbar legend and text descriptors for comfort bands
    cb = fig.colorbar(
        heatmap,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=UTCI_BOUNDS[1:-1],
        drawedges=False,
    )
    cb.outline.set_visible(False)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")

    # Add labels to the colorbar
    for label, position in list(zip(*[UTCI_CATEGORIES, np.linspace(0, 1, 21)[1::2]])):
        colorbar_ax.text(
            position,
            1,
            textwrap.fill(label, 14),
            ha="center",
            va="bottom",
            size="small",
            transform=colorbar_ax.transAxes,
        )

    # Add stacked plot
    t = pd.cut(series, UTCI_BOUNDS, labels=UTCI_CATEGORIES)
    t = t.groupby([t.index.month, t]).count().unstack().T
    t = t / t.sum()
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    t.T.plot.bar(
        ax=histogram_ax,
        stacked=True,
        color=UTCI_COLOURS,
        legend=False,
        width=1,
    )
    histogram_ax.set_xlabel(None)
    histogram_ax.set_xlim(-0.5, 11.5)
    histogram_ax.set_ylim(0, 1)
    histogram_ax.set_xticklabels(months, ha="center", rotation=0, color="k")
    plt.setp(histogram_ax.get_yticklabels(), color="k")
    for spine in ["top", "right"]:
        histogram_ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        histogram_ax.spines[spine].set_color("k")
    histogram_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

    # # Add header percentages for bar plot
    cold_stress = t.T.filter(regex="cold").sum(axis=1)
    heat_stress = t.T.filter(regex="heat").sum(axis=1)
    no_stress = 1 - cold_stress - heat_stress
    for n, (i, j, k) in enumerate(zip(*[cold_stress, no_stress, heat_stress])):
        histogram_ax.text(
            n,
            1.02,
            "{0:0.1f}%".format(i * 100),
            va="bottom",
            ha="center",
            color=UTCICategory.moderate_cold_stress.hex_colour,
            fontsize="small",
        )
        histogram_ax.text(
            n,
            1.02,
            "{0:0.1f}%\n".format(j * 100),
            va="bottom",
            ha="center",
            color=UTCICategory.no_thermal_stress.hex_colour,
            fontsize="small",
        )
        histogram_ax.text(
            n,
            1.02,
            "{0:0.1f}%\n\n".format(k * 100),
            va="bottom",
            ha="center",
            color=UTCICategory.strong_heat_stress.hex_colour,
            fontsize="small",
        )

    if title is None:
        heatmap_ax.set_title(series.name, color="k", y=1, ha="left", va="bottom", x=0)
    else:
        heatmap_ax.set_title(
            f"{series.name}\n{title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    return fig
