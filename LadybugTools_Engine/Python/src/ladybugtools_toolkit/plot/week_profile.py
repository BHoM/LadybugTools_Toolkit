from typing import Tuple, Union

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def week_profile(
    series: pd.Series,
    color: Union[str, Tuple] = "k",
    title: str = None,
) -> Figure:
    """Plot a profile aggregated across days of week in a given Series.

    Args:
        series (pd.Series):
            A time-indexed Pandas Series object.
        color (Union[str, Tuple], optional):
            The color to use for this plot.
        ylabel (str, optional):
            A label to be placed on the y-axis.
        title (str, optional):
            A title to place at the top of the plot. Defaults to None.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series passed is not datetime indexed.")

    minmax_range = [0.0001, 0.9999]
    q_range = [0.05, 0.95]
    color = "slategrey"
    q_alpha = 0.3
    minmax_alpha = 0.1

    # Remove outliers
    series = series[
        (series >= series.quantile(minmax_range[0]))
        & (series <= series.quantile(minmax_range[1]))
    ]

    # remove nan/inf
    series = series.replace(-np.inf, np.nan).replace(np.inf, np.nan).dropna()

    start_idx = series.dropna().index.min()
    end_idx = series.dropna().index.max()

    # group data
    group = series.groupby([series.index.dayofweek, series.index.time])

    # count n samples per timestep
    n_samples = group.count().mean()

    # Get groupped data
    minima = group.min()
    lower = group.quantile(q_range[0])
    median = group.median()
    mean = group.mean()
    upper = group.quantile(q_range[1])
    maxima = group.max()

    # create df for re-indexing
    df = pd.concat([minima, lower, median, mean, upper, maxima], axis=1)
    df.columns = ["minima", "lower", "median", "mean", "upper", "maxima"]
    df = df.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(0)

    # reset index and rename
    df = df.reset_index()
    idx = []
    for day, hour, minute in list(
        zip(
            *[
                [i + 1 for i in df.level_0.values],
                [i.hour for i in df.level_1.values],
                [i.minute for i in df.level_1.values],
            ]
        )
    ):
        idx.append(pd.to_datetime(f"2007-01-{day:02d} {hour:02d}:{minute:02d}:00"))
    df.index = idx
    df.drop(columns=["level_0", "level_1"], inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # q-q
    ax.fill_between(
        df.index,
        df["lower"],
        df["upper"],
        alpha=q_alpha,
        color=color,
        lw=None,
        ec=None,
        label=f"{q_range[0]:0.0%}ile-{q_range[1]:0.0%}ile",
    )
    # q-extreme
    ax.fill_between(
        df.index,
        df["lower"],
        df["minima"],
        alpha=minmax_alpha,
        color=color,
        lw=None,
        ec=None,
        label="min-max",
    )
    ax.fill_between(
        df.index,
        df["upper"],
        df["maxima"],
        alpha=minmax_alpha,
        color=color,
        lw=None,
        ec=None,
        label="_nolegend_",
    )
    # mean/median
    ax.plot(df.index, df["mean"], c=color, ls="-", lw=1, label="Mean")
    ax.plot(df.index, df["median"], c=color, ls="--", lw=1, label="Median")

    # format axes
    ax.set_xlim(df.index.min(), df.index.max())
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("k")

    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.2)
    ax.grid(visible=True, which="minor", axis="both", c="k", ls=":", lw=1, alpha=0.1)
    ax.xaxis.set_major_formatter(md.DateFormatter("%a %H:%M"))
    ax.xaxis.set_minor_locator(md.HourLocator(byhour=[0, 6, 12, 18]))

    # legend
    lgd = ax.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc=8,
        ncol=6,
        borderaxespad=0,
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    for text in lgd.get_texts():
        plt.setp(text, color="k")

    ti = f"Typical week between {start_idx:%Y-%m-%d} and {end_idx:%Y-%m-%d} (~{n_samples:0.1f} samples per timestep)"
    if title is not None:
        ti += "\n" + title
    ax.set_title(
        ti,
        ha="left",
        x=0,
    )

    plt.tight_layout()

    return fig
