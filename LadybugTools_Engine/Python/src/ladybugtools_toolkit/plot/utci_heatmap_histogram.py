import calendar
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..ladybug_extension.datacollection import collection_to_series
from . import UTCI_BOUNDARYNORM, UTCI_COLORMAP, UTCI_LABELS, UTCI_LEVELS


def utci_heatmap_histogram(
    collection: HourlyContinuousCollection,
    **kwargs,
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )

    title = kwargs.get("title", None)

    # Instantiate figure
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[4, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.75)

    # Construct series
    series = collection_to_series(collection)

    # Add heatmap
    heatmap = heatmap_ax.imshow(
        pd.pivot_table(
            series.to_frame(),
            index=series.index.time,
            columns=series.index.date,
            values=series.name,
        ).values[::-1],
        norm=UTCI_BOUNDARYNORM,
        extent=[
            mdates.date2num(series.index.min()),
            mdates.date2num(series.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        cmap=UTCI_COLORMAP,
        interpolation="none",
    )

    heatmap_ax.xaxis_date()
    heatmap_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    heatmap_ax.yaxis_date()
    heatmap_ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    heatmap_ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(heatmap_ax.get_xticklabels(), ha="left", color="k")
    plt.setp(heatmap_ax.get_yticklabels(), color="k")
    for spine in ["top", "bottom", "left", "right"]:
        heatmap_ax.spines[spine].set_visible(False)
        heatmap_ax.spines[spine].set_color("k")
    heatmap_ax.grid(visible=True, which="major", color="k", linestyle=":", alpha=0.5)

    # Add colorbar legend and text descriptors for comfort bands
    cb = fig.colorbar(
        heatmap,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=UTCI_LEVELS,
        drawedges=False,
    )
    cb.outline.set_visible(False)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")

    # Add labels to the colorbar
    levels = [-100] + UTCI_LEVELS + [100]
    for n, ((low, high), label) in enumerate(
        zip(*[list(zip(levels[:-1], levels[1:])), UTCI_LABELS])
    ):
        if n == 0:
            ha = "right"
            position = high
        elif n == len(levels) - 2:
            ha = "left"
            position = low
        else:
            ha = "center"
            position = (low + high) / 2

        colorbar_ax.text(
            position,
            1,
            textwrap.fill(label, 9),
            ha=ha,
            va="bottom",
            size="small",
            # transform=colorbar_ax.transAxes,
        )

    # Add stacked plot
    t = pd.cut(series, [-100] + UTCI_LEVELS + [100], labels=UTCI_LABELS)
    t = t.groupby([t.index.month, t]).count().unstack().T
    t = t / t.sum()
    months = [calendar.month_abbr[i] for i in range(1, 13, 1)]
    t.T.plot.bar(
        ax=histogram_ax,
        stacked=True,
        color=[rgb2hex(UTCI_COLORMAP.get_under())]
        + UTCI_COLORMAP.colors
        + [rgb2hex(UTCI_COLORMAP.get_over())],
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
    histogram_ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    # # Add header percentages for bar plot
    cold_stress = t.T.filter(regex="Cold").sum(axis=1)
    heat_stress = t.T.filter(regex="Heat").sum(axis=1)
    no_stress = 1 - cold_stress - heat_stress
    for n, (i, j, k) in enumerate(zip(*[cold_stress, no_stress, heat_stress])):
        histogram_ax.text(
            n,
            1.02,
            f"{i:0.1%}",
            va="bottom",
            ha="center",
            color="#3C65AF",
            fontsize="small",
        )
        histogram_ax.text(
            n,
            1.02,
            f"{j:0.1%}\n",
            va="bottom",
            ha="center",
            color="#2EB349",
            fontsize="small",
        )
        histogram_ax.text(
            n,
            1.02,
            f"{k:0.1%}\n\n",
            va="bottom",
            ha="center",
            color="#C31F25",
            fontsize="small",
        )

    if title is None:
        heatmap_ax.set_title(series.name, color="k", y=1, ha="left", va="bottom", x=0)
    else:
        heatmap_ax.set_title(
            f"{series.name} - {title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    return fig
