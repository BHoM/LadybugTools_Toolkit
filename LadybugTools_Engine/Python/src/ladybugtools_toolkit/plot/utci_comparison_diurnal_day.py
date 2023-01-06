import calendar
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure

from ..ladybug_extension.datacollection import to_series
from .colormaps import UTCI_COLORMAP, UTCI_LABELS, UTCI_LEVELS
from .lighten_color import lighten_color


def utci_comparison_diurnal_day(
    collections: List[HourlyContinuousCollection],
    month: int = 6,
    collection_ids: List[str] = None,
    agg: str = "mean",
    title: str = None,
) -> Figure:
    """Plot a set of UTCI collections on a single figure for monthly diurnal periods.

    Args:
        collections (List[HourlyContinuousCollection]):
            A list of UTCI collections.
        month (int, optional):
            The month to get the typical day from. Default is 6.
        collection_ids (List[str], optional):
            A list of descriptions for each of the input collections. Defaults to None.
        agg (str, optional):
            How to generate the "typical" day. Defualt is "mean" which uses the mean for each timestep in that month.
        title (str, optional):
            A custom title to add to this plot.

    Returns:
        Figure:
            A matplotlib figure object.
    """

    if agg not in ["min", "mean", "max", "median"]:
        raise ValueError("agg is not of a possible type.")

    if collection_ids is None:
        collection_ids = [f"{i:02d}" for i in range(len(collections))]
    assert len(collections) == len(
        collection_ids
    ), "The length of collections_ids must match the number of collections."

    for n, col in enumerate(collections):
        if not isinstance(col.header.data_type, UniversalThermalClimateIndex):
            raise ValueError(
                f"Collection {n} data type is not UTCI and cannot be used in this plot."
            )

    # combine utcis and add names to columns
    df = pd.concat([to_series(i) for i in collections], axis=1, keys=collection_ids)
    ylims = df.min().min(), df.max().max()
    df_agg = df.groupby([df.index.month, df.index.hour], axis=0).agg(agg).loc[month]
    df_agg.index = range(24)
    # add a final value to close the day
    df_agg.loc[24] = df_agg.loc[0]

    fig, ax = plt.subplots(1, 1)

    for col in df_agg.columns:
        ax.plot(
            df_agg[col].index,
            df_agg[col].values,
            lw=1,
            label=col,
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")

    # Fill between ranges
    utci_handles = []
    utci_labels = []
    for low, high, color, category in list(
        zip(
            *[
                ([-100] + UTCI_LEVELS + [100])[0:-1],
                ([-100] + UTCI_LEVELS + [100])[1:],
                [rgb2hex(UTCI_COLORMAP.get_under())]
                + UTCI_COLORMAP.colors
                + [rgb2hex(UTCI_COLORMAP.get_over())],
                UTCI_LABELS,
            ]
        )
    ):
        cc = lighten_color(color, 0.2)
        ax.axhspan(low, high, color=cc)
        # Get fille color attributes
        utci_labels.append(category)
        utci_handles.append(mpatches.Patch(color=cc, label=category))

    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.1)
    ax.grid(visible=True, which="minor", axis="x", c="k", ls=":", lw=1, alpha=0.1)

    # get handles
    mitigation_handles, mitigation_labels = ax.get_legend_handles_labels()

    # Format plots
    ax.set_xlim(0, 24)
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 6, 12, 18]))
    ax.xaxis.set_minor_locator(plt.FixedLocator([3, 9, 15, 21]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00"], minor=False, ha="left")

    # Construct legend
    handles = utci_handles + mitigation_handles
    labels = utci_labels + mitigation_labels
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=[1, 0.9],
        frameon=False,
        fontsize="small",
        ncol=1,
    )

    ti = f"{calendar.month_name[month]} typical day ({agg})"
    if title is not None:
        ti = f"{ti}\n{title}"
    ax.set_title(
        ti,
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

    return fig
