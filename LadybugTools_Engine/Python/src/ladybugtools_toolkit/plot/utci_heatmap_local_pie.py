import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.utci_pie_local import utci_pie_local
from ladybugtools_toolkit.plot.colormaps_local import (
    UTCI_LOCAL_BOUNDARYNORM,
    UTCI_LOCAL_BOUNDARYNORM_IP,
    UTCI_LOCAL_COLORMAP,
    UTCI_LOCAL_LABELS,
    UTCI_LOCAL_LEVELS,
    UTCI_LOCAL_LEVELS_IP,
)
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch


def utci_heatmap_local_pie(
    collection: HourlyContinuousCollection, title: str = None, analysis_period: AnalysisPeriod = AnalysisPeriod(), show_legend: bool = True, IP: bool = True
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        title (str, optional):
            A title to add to the resulting figure. Default is None.
        analysis_period (AnalysisPeriod, optional):
            A ladybug analysis period.
        show_legend (bool, optional):
            Set to True to plot the legend also. Default is True.
        IP (bool, optional):
            Convert data to IP unit. Default is True.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )

    # Convert data to IP unit
    if IP:
        collection = collection.to_ip()

    # Instantiate figure
    fig = plt.figure(figsize=(15, 4), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=2, nrows=2, width_ratios=[5, 1], height_ratios=[1, 0.03], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    pie_ax = fig.add_subplot(spec[0, 1])
    colorbar_ax = fig.add_subplot(spec[1, :])

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
        norm=UTCI_LOCAL_BOUNDARYNORM_IP if IP else UTCI_LOCAL_BOUNDARYNORM,
        extent=[
            mdates.date2num(series.index.min()),
            mdates.date2num(series.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        cmap=UTCI_LOCAL_COLORMAP,
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

    # Add pie chart
    series_adjust = to_series(collection.filter_by_analysis_period(analysis_period))

    series_cut = pd.cut(series_adjust, bins=[-100] + UTCI_LOCAL_LEVELS_IP + [200] if IP else [-100] + UTCI_LOCAL_LEVELS + [200], labels=UTCI_LOCAL_LABELS)
    sizes = (series_cut.value_counts() / len(series_adjust))[UTCI_LOCAL_LABELS]
    colors = (
        [UTCI_LOCAL_COLORMAP.get_under()] + UTCI_LOCAL_COLORMAP.colors + [UTCI_LOCAL_COLORMAP.get_over()]
    )

    pie_ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
        radius=1.5,
    )

    centre_circle = plt.Circle((0, 0), 0.80, fc="white")
    pie_ax.add_artist(centre_circle)

    if show_legend:
        # construct custom legend including values
        legend_elements = [
            Patch(
                facecolor=color, edgecolor=None, label=f"[{sizes[label]:02.0%}]"
            )
            for label, color in list(zip(*[UTCI_LOCAL_LABELS[5:7], colors[5:7]]))
        ]
        lgd = pie_ax.legend(handles=legend_elements, loc="center", frameon=False)
        lgd.get_frame().set_facecolor((1, 1, 1, 0))
    
    # Add colorbar legend and text descriptors for comfort bands
    cb = fig.colorbar(
        heatmap,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=UTCI_LOCAL_LEVELS_IP if IP else UTCI_LOCAL_LEVELS,
        drawedges=False,
    )
    cb.outline.set_visible(False)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")

    # Add labels to the colorbar
    levels = [-100] + UTCI_LOCAL_LEVELS_IP + [200] if IP else [-100] + UTCI_LOCAL_LEVELS + [200]
    for n, ((low, high), label) in enumerate(
        zip(*[[(x, y) for x, y in zip(levels[:-1], levels[1:])], UTCI_LOCAL_LABELS])
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
