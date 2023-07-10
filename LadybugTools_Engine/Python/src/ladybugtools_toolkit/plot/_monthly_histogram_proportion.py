import calendar
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from ..helpers import contrasting_color, validate_timeseries


def monthly_histogram_proportion(
    series: pd.Series,
    bins: List[float],
    ax: plt.Axes = None,
    labels: List[str] = None,
    show_year_in_label: bool = False,
    show_labels: bool = False,
    **kwargs,
) -> plt.Axes:
    """Create a monthly histogram of a pandas Series.

    Args:
        series (pd.Series):
            The pandas Series to plot. Must have a datetime index.
        bins (List[float]):
            The bins to use for the histogram.
        ax (plt.Axes, optional):
            An optional plt.Axes object to populate. Defaults to None, which creates a new plt.Axes object.
        labels (List[str], optional):
            The labels to use for the histogram. Defaults to None, which uses the bin edges.
        show_year_in_label (bool, optional):
            Whether to show the year in the x-axis label. Defaults to False.
        show_labels (bool, optional):
            Whether to show the labels on the bars. Defaults to False.
        **kwargs:
            Additional keyword arguments to pass to plt.bar.

    Returns:
        plt.Axes:
            The populated plt.Axes object.
    """

    validate_timeseries(series)

    if ax is None:
        ax = plt.gca()

    t = pd.cut(series, bins=bins, labels=labels)
    t = t.groupby([t.index.year, t.index.month, t]).count().unstack().T
    t = t / t.sum()

    # adjust column labels
    if show_year_in_label:
        t.columns = [
            f"{year}\n{calendar.month_abbr[month]}" for year, month in t.columns.values
        ]
    else:
        t.columns = [f"{calendar.month_abbr[month]}" for _, month in t.columns.values]

    t.T.plot.bar(
        ax=ax,
        stacked=True,
        legend=False,
        width=1,
        **kwargs,
    )
    ax.set_xlim(-0.5, len(t.columns) - 0.5)
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), ha="center", rotation=0)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    if show_labels:
        for i, c in enumerate(ax.containers):
            label_colors = [contrasting_color(i.get_facecolor()) for i in c.patches]
            labels = [
                f"{v.get_height():0.1%}" if v.get_height() > 0.1 else "" for v in c
            ]
            ax.bar_label(
                c,
                labels=labels,
                label_type="center",
                color=label_colors[i],
                fontsize="xx-small",
            )

    return ax
