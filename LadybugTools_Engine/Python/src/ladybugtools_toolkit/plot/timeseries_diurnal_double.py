from typing import List, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.figure import Figure


from ladybugtools_toolkit import analytics


# @analytics
def timeseries_diurnal_double(
    series1: pd.Series,
    series2: pd.Series,
    color1: Union[str, Tuple] = "r",
    color2: Union[str, Tuple] = "k",
    ylabel: str = None,
    title: str = None,
    ylims: List[float] = None,
) -> Figure:
    """Plot a heatmap for a Pandas Series object.

    Args:
        series1 (pd.Series):
            A time-indexed Pandas Series object.
        series2 (pd.Series):
            A time-indexed Pandas Series object.
        color1 (Union[str, Tuple], optional):
            The color to use for series1 diurnal plot.
        color2 (Union[str, Tuple], optional):
            The color to use for series2 diurnal plot.
        ylabel (str, optional):
            A label to be placed on the y-axis.
        title (str, optional):
            A title to place at the top of the plot. Defaults to None.
        ylims (List[float], optional):
            Set the y-limits for the plot.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(series1.index, pd.DatetimeIndex):
        raise ValueError("Series 1 passed is not datetime indexed.")
    
    target_index = pd.MultiIndex.from_arrays(
        [
            [x for xs in [[i + 1] * 24 for i in range(12)] for x in xs],
            [x for xs in [range(0, 24, 1) for i in range(12)] for x in xs],
        ],
        names=["month", "hour"],
    )

    # groupby and obtain min, quant, mean, quant, max values
    grp1 = series1.groupby([series1.index.month, series1.index.hour], axis=0)
    _min = grp1.min().reindex(target_index)
    _lower = grp1.quantile(0.05).reindex(target_index)
    _mean = grp1.mean().reindex(target_index)
    _upper = grp1.quantile(0.95).reindex(target_index)
    _max = grp1.max().reindex(target_index)

    grp2 = series2.groupby([series2.index.month, series2.index.hour], axis=0)
    _mean2 = grp2.mean().reindex(target_index)

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    x_values = range(288)

    # for each month, plot the diurnal profile
    for i in range(0, 288)[::24]:
        ax.plot(
            x_values[i : i + 24],
            _mean[i : i + 24],
            color=color1,
            lw=2,
            label="Canyon Average",
            zorder=7,
        )
        ax.plot(
            x_values[i : i + 24],
            _lower[i : i + 24],
            color=color1,
            lw=1,
            label="Canyon Average",
            ls=":",
        )
        ax.plot(
            x_values[i : i + 24],
            _upper[i : i + 24],
            color=color1,
            lw=1,
            label="Canyon Average",
            ls=":",
        )
        ax.plot(
            x_values[i : i + 24],
            _mean2[i : i + 24],
            color=color2,
            lw=2,
            label="DBT Average",
        )
        ax.fill_between(
            x_values[i : i + 24],
            _min[i : i + 24],
            _max[i : i + 24],
            color=color1,
            alpha=0.2,
            label="Range",
        )
        ax.fill_between(
            x_values[i : i + 24],
            _lower[i : i + 24],
            _upper[i : i + 24],
            color="white",
            alpha=0.5,
            label="Range",
        )
        
        ax.set_ylabel(
            series1.name,
            labelpad=2,
        )

    ax.xaxis.set_major_locator(mticker.FixedLocator(range(0, 288, 24)))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(range(12, 288, 24)))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(7))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")

    ax.set_xlim([0, 287])
    ax.set_xticklabels(
        [
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
        ],
        minor=False,
        ha="left",
        color="k",
    )
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
    ax.grid(visible=True, which="minor", axis="both", c="k", ls=":", lw=1, alpha=0.1)

    handles = [
        mlines.Line2D([0], [0], label="Canyon Average", color=color1, lw=2),
        mlines.Line2D([0], [0], label="EPW Average", color=color2, lw=2),
        mlines.Line2D([0], [0], label="5-95%ile", color=color1, lw=1, ls=":"),
        mpatches.Patch(color=color1, label="Range", alpha=0.3),
    ]

    lgd = ax.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.2),
        loc=8,
        ncol=6,
        borderaxespad=0,
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    for text in lgd.get_texts():
        plt.setp(text, color="k")

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if ylims is not None:
        ax.set_ylim(ylims)

    if title is None:
        ax.set_title(
            f"Monthly average diurnal profile\n{series1.name}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )
    else:
        ax.set_title(f"{title}", color="k", y=1, ha="left", va="bottom", x=0)

    plt.tight_layout()

    return fig
