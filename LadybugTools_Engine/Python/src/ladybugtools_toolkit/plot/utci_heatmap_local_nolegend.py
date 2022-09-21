import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from ladybugtools_toolkit.plot.colormaps_local import (
    UTCI_LOCAL_BOUNDARYNORM,
    UTCI_LOCAL_COLORMAP,
    UTCI_LOCAL_LABELS,
    UTCI_LOCAL_LEVELS,
)
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def utci_heatmap_local_nolegend(
    collection: HourlyContinuousCollection, title: str = None
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        title (str, optional):
            A title to add to the resulting figure. Default is None.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )

    # Instantiate figure
    fig = plt.figure(figsize=(15, 3.29), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=1, width_ratios=[1], height_ratios=[1], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
 
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
        norm=UTCI_LOCAL_BOUNDARYNORM,
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
