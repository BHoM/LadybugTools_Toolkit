import textwrap
import matplotlib as mpl

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug.analysisperiod import AnalysisPeriod
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.colormaps import (
    UTCI_BOUNDARYNORM,
    UTCI_BOUNDARYNORM_IP,
    UTCI_COLORMAP,
    UTCI_LABELS,
    UTCI_LEVELS,
    UTCI_LEVELS_IP,
)
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def utci_heatmap_masked_stacked(
    collection: HourlyContinuousCollection, title: str = None, show_legend: bool = True, IP: bool = True, masked: bool = True, analysis_period: AnalysisPeriod = AnalysisPeriod()
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        title (str, optional):
            A title to add to the resulting figure. Default is None.
        show_legend (bool, optional):
            Set to True to plot the legend. Default is True.
        IP (bool, optional):
            Convert data to IP unit. Default is True.
        masekd (bool, optional):
            Set to True to mask UTCI with ananlysis period. Default is True.
        analysis_period (AnalysisPeriod, optional):
            A ladybug analysis period.

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
        ncols=2, nrows=2, width_ratios=[5, 0.75], height_ratios=[1, 0.03], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    bar_ax = fig.add_subplot(spec[0, 1])
    # divider = make_axes_locatable(heatmap_ax)
    # if show_legend:
    #     colorbar_ax = divider.append_axes("bottom", size="3%", pad=0.85)
    if show_legend:
        colorbar_ax = fig.add_subplot(spec[1, :])

    # Construct series
    series = to_series(collection)
    data = pd.pivot_table(
                series.to_frame(),
                index=series.index.time,
                columns=series.index.date,
                values=series.name,
            ).values[::-1]
    
    
    # Add heatmap
    heatmap = heatmap_ax.imshow(
        data,
        norm=UTCI_BOUNDARYNORM_IP if IP else UTCI_BOUNDARYNORM,
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
    mask = np.copy(data)

    if masked:
        st_hour = analysis_period.st_hour
        end_hour = analysis_period.end_hour
        st_day = analysis_period.doys_int[0] - 1
        end_day = analysis_period.doys_int[-1] - 1
        end_count = 23 - end_hour - 1
        start_count = 23 - st_hour

        for i in range (0,len(data)):
            if (i <= end_count or i > start_count):
                mask[i] = np.ones(len(data[i])) 
            else:
                mask[i] = np.ones(len(data[i])) * (-1)
                for j in range (0,len(mask[i])):
                    if (j < st_day or j > end_day):
                        mask[i][j] = 1

        # Set masked colormap
        maskedCM = mpl.cm.get_cmap("gray").copy()
        maskedCM.set_over(color='k',alpha=0.5)
        maskedCM.set_under(color='w', alpha=0)
        maskedNorm = BoundaryNorm([0,0.5],maskedCM.N)
    else:
        maskedNorm = UTCI_BOUNDARYNORM_IP
        maskedCM = UTCI_COLORMAP
        
    heatmap_ax.imshow(
        mask, 
        norm=maskedNorm,
        extent=[
            mdates.date2num(series.index.min()),
            mdates.date2num(series.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        cmap = maskedCM,
        interpolation='none',
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

    # Add stacked bar chart
    series_adjust = to_series(collection.filter_by_analysis_period(analysis_period))
    series_cut = pd.cut(series_adjust, bins=[-100] + UTCI_LEVELS_IP + [200] if IP else [-100] + UTCI_LEVELS + [200], labels=UTCI_LABELS)
    sizes = (series_cut.value_counts() / len(series_adjust))[UTCI_LABELS]
    colors = (
        [UTCI_COLORMAP.get_under()] + UTCI_COLORMAP.colors + [UTCI_COLORMAP.get_over()]
    )

    catagories = sizes.index.tolist()

    for i in range(len(sizes)):
        # print(catagories[ind],cata)
        val = sizes[i]*100
        # if (val) != 0:
            # print(catagories[i], ": ", val)

        bar_ax.bar('Roof',val,bottom=np.sum(sizes[:i]*100, axis = 0),color=colors[i], label=catagories[i])
        # ax.legend()
        bar_ax.axis('off')

    # Add text to each bar patch
    for i in range(len(bar_ax.patches)):
        if round(sizes[i]*100,0) > 1:
            # print(round(sizes[i]*100,0))
            bar_ax.text(bar_ax.patches[i].get_x() + bar_ax.patches[i].get_width() / 2,
            bar_ax.patches[i].get_height() / 2 + bar_ax.patches[i].get_y(),
            str(int(round(sizes[i]*100,0))) + "%", ha = 'center',
            color = 'k', size = 8)

    if show_legend:
        # Add colorbar legend and text descriptors for comfort bands
        cb = fig.colorbar(
            heatmap,
            cax=colorbar_ax,
            orientation="horizontal",
            ticks=UTCI_LEVELS_IP if IP else UTCI_LEVELS,
            drawedges=False,
        )
        cb.outline.set_visible(False)
        plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")

        # Add labels to the colorbar
        levels = [-100] + UTCI_LEVELS_IP + [200] if IP else [-100] + UTCI_LEVELS + [200]
        for n, ((low, high), label) in enumerate(
            zip(*[[(x, y) for x, y in zip(levels[:-1], levels[1:])], UTCI_LABELS])
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
    # Add title to the pie
    if st_hour == 23:
        st_hour_title = str(st_hour - 12) + " am"
    elif st_hour < 12:
        st_hour_title = str(st_hour) + " am"
    else:
        st_hour_title = str(st_hour - 12) + " pm"

    if end_hour == 23:
        end_hour_title = str(end_hour - 12 + 1) + "am"
    elif end_hour < 12:
        end_hour_title = str(end_hour + 1) + "am"
    else:
        end_hour_title = str(end_hour - 12 + 1) + " pm"
    bar_title = "Occupied hours:" + st_hour_title + " to " + end_hour_title

    if title is None:
        heatmap_ax.set_title(series.name, color="k", y=1, ha="left", va="bottom", x=0)
        bar_ax.set_title("Design focus period: full day", color="k", size="small")

    else:
        heatmap_ax.set_title(
            f"{series.name} - {title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )
        bar_ax.set_title(bar_title, color="k", size="small")
    return fig
