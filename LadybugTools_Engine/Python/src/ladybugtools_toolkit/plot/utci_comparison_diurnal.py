from typing import Any, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure

from ..helpers import lighten_color
from . import UTCI_COLORMAP, UTCI_LABELS, UTCI_LEVELS


def utci_comparison_diurnal(
    collections: List[HourlyContinuousCollection],
    collection_ids: List[str] = None,
    colors: List[Any] = None,
) -> Figure:
    """Plot a set of UTCI collections on a single figure for monthly diurnal periods.

    Args:
        collections (List[HourlyContinuousCollection]):
            A list of UTCI collections.
        collection_ids (List[str], optional):
            A list of descriptions for each of the input collections. Defaults to None.
        colors (List[Any], optional):
            A list of colors to use for the lines. Defaults to None which uses the cycler to determine which colors to use.

    Returns:
        Figure:
            A matplotlib figure object.
    """

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

    if colors is not None:
        if len(colors) != len(collections):
            raise ValueError(
                "The number of colors must match the number of collections."
            )
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True, sharey=True)
    for n, ax in enumerate(axes.flat):
        for nn, col in enumerate(collections):
            values = col.average_monthly_per_hour().values
            ax.plot(
                range(25),
                list(values[n * 24 : (n * 24) + 24]) + [values[n * 24]],
                lw=1,
                label=collection_ids[nn],
            )
            ax.set_title(months[n], x=0, ha="left")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("k")

    # Get plotted values attributes
    ylim = axes.flat[-1].get_ylim()
    mitigation_handles, mitigation_labels = axes.flat[-1].get_legend_handles_labels()

    # Fill between ranges
    for n, ax in enumerate(axes.flat):
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
        ax.grid(
            visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.1
        )
        ax.grid(visible=True, which="minor", axis="x", c="k", ls=":", lw=1, alpha=0.1)
        if n in [0, 3, 6, 9]:
            ax.set_ylabel("UTCI (C)")
        if n in [9, 10, 11]:
            ax.set_xlabel("Time of day")

    # Format plots
    ax.set_xlim(0, 24)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 6, 12, 18]))
    ax.xaxis.set_minor_locator(plt.FixedLocator([3, 9, 15, 21]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00"], minor=False, ha="left")

    # Construct legend
    handles = utci_handles + mitigation_handles
    labels = utci_labels + mitigation_labels
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=[1, 0.9],
        frameon=False,
        fontsize="small",
        ncol=1,
    )

    fig.suptitle(
        "Average diurnal profile",
        ha="left",
        va="bottom",
        x=0.05,
        y=0.95,
    )

    plt.tight_layout()

    return fig
