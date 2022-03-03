import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import calendar
import colorsys
import textwrap
from typing import List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from external_comfort.plot.colours import (
    UTCI_BOUNDS,
    UTCI_CATEGORIES,
    UTCI_COLOURMAP,
    UTCI_COLOURMAP_NORM,
    UTCI_COLOURS,
    UTCICategory,
)
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug_extension.datacollection import from_series, to_series
from matplotlib import patches
from matplotlib.colors import cnames, to_rgb
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline

WIDTH = 15
HEIGHT = 5

pd.plotting.register_matplotlib_converters()

import re


def _lighten_color(color: str, amount: float = 0.5) -> Tuple[float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*to_rgb(c)))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def _camel_case_split(string: str) -> str:
    """Add spaces between distinct words in a CamelCase formatted string."""
    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", string
    )
    return " ".join([m.group(0) for m in matches])


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


def utci_pseudo_journey(
    utci_collections: List[HourlyContinuousCollection],
    month: int,
    hour: int,
    names: List[str] = None,
    curve: bool = False,
    ylims: List[float] = None,
    show_legend: bool = True,
    fig_width: float = 10,
    fig_height: float = 5
) -> Figure:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a given time of year

    Args:
        utci_collections (HourlyContinuousCollection): A list of UTCI HourlyContinuousCollection objects
        month (int): The month of the year to plot
        hour (int): The hour of the day to plot
        names (List[str], optional): A list of names to label each condition with. Defaults to None.
        curve (bool, optional): Whether to plot the pseudo-journey as a spline. Defaults to False.
        ylims (List[float], optional): A list of y-axis limits to use. Defaults to None.

    Returns:
        Figure: A matplotlib figure object
    """

    # Check that all collections are UTCI
    for collection in utci_collections:
        if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
            raise ValueError(
                "Collection data type is not UTCI and cannot be used in this plot."
            )

    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12.")

    if not 0 <= hour <= 23:
        raise ValueError("Hour must be between 0 and 23.")
    
    if not isinstance(ylims, (None, list)) or len(ylims) != 2:
        raise ValueError("ylims must be a list of length 2.")

    if names:
        if len(utci_collections) != len(names):
            raise ValueError("Number of collections and names must be equal.")

    # Convert collections into series and combine
    df = pd.concat([to_series(col) for col in utci_collections], axis=1, keys=names)
    df_pit = df[(df.index.month == month) & (df.index.hour == hour)].mean()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        name = _camel_case_split(idx).replace(" ", "\n")
        ax.text(
            n, val, name, c="k", zorder=10, ha="center", va="center", fontsize="medium"
        )

    if ylims is None:
        ylims = ax.get_ylim()

    if curve:
        # Smooth values
        if len(utci_collections) < 3:
            k = 1
        else:
            k = 2
        x = np.arange(len(utci_collections))
        y = df_pit.values
        xnew = np.linspace(min(x), max(x), 300)
        bspl = make_interp_spline(x, y, k=k)
        ynew = bspl(xnew)

        # Plot the smoothed values
        ax.plot(xnew, ynew, c="#B30202", ls="--", zorder=3)

    [ax.spines[spine].set_visible(False) for spine in ["top", "right", "bottom"]]
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    # Add colors to plot
    utci_handles = []
    utci_labels = []
    for low, high, color, category in list(
        zip(
            *[
                UTCI_BOUNDS[0:-1],
                UTCI_BOUNDS[1:],
                UTCI_COLOURS,
                UTCI_CATEGORIES,
            ]
        )
    ):
        cc = _lighten_color(color, 0.2)
        ax.axhspan(low, high, color=cc)
        utci_labels.append(category)
        utci_handles.append(patches.Patch(color=cc, label=category))

    if len(ylims) == 1:
        _middf = 0.5 * (df_pit.max() + df_pit.min())
        ax.set_ylim([_middf - 1 - ylims[0], _middf + 1 +ylims[0]])
    elif len(ylims) == 2:
        ax.set_ylim([ylims[0] - 1, ylims[1] + 1])
    
    ax.set_xlim(-0.5, len(utci_collections) - 0.5)

    if show_legend:
        lgd = fig.legend(
            utci_handles,
            utci_labels,
            bbox_to_anchor=(1, 0.9),
            loc="upper left",
            ncol=1,
            borderaxespad=0,
            frameon=False,
        )
        lgd.get_frame().set_facecolor((1, 1, 1, 0))
        [plt.setp(text, color="k") for text in lgd.get_texts()]

    ax.set_title(
        f"{calendar.month_name[month]} {hour:02d}:00", x=0, ha="left", va="bottom"
    )

    #ax.set_ylim(ylims[0] - 1, ylims[1] + 1)

    ax.set_ylabel("UTCI (°C)")
    plt.tight_layout()

    return fig

if __name__ =="__main__":
    pass
