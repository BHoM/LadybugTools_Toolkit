from __future__ import annotations

from typing import List, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, ListedColormap
from scipy import stats
from scipy.stats import exponweib

from ..helpers import rolling_window, weibull_pdf
from .direction_bins import DirectionBins


def cumulative_probability(
    wind_speeds: List[float],
    bins: List[float] = None,
    percentiles: List[float] = None,
    title: str = None,
) -> plt.Figure:
    """Plot a cumulative probability graph, showing binned wind speeds in a cumulative
        frequency histogram.

    Args:
        wind_speeds (List[float]):
            A list of wind speeds.
        bins (List[float], optional):
            A set of bin edges to categorise the wind speeds into. Defaults to None.
        percentiles (List[float], optional):
            A list of percentiles to show on the chart. Defaults to None.
        title (str, optional):
            A title to add to the resulting plot. Defaults to None.

    Returns:
        plt.Figure:
            A Figure object.
    """
    if percentiles is None:
        percentiles = [0.5, 0.95]
    if (min(percentiles) < 0) or (max(percentiles) > 1):
        raise ValueError("percentiles must fall within the range 0-1.")

    if bins is None:
        bins = np.linspace(0, 25, 50)
    if min(bins) < 0:
        raise ValueError("Minimum bin value must be >= 0")

    x = bins
    y = [stats.percentileofscore(wind_speeds, i) / 100 for i in bins]

    # remove all but one element from lists where probability is == 1
    if len([i for i in y if i == 1]) > 2:
        idxmax = np.where([i == 1 for i in y])[0][0]
        x = x[: idxmax + 1]
        y = y[: idxmax + 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x, y, c="grey")
    ax.set_xlim(0, max(x))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    for percentile in percentiles:
        val = np.quantile(wind_speeds, percentile)
        ax.hlines(percentile, 0, val, ls="--", lw=1, colors="k")
        ax.vlines(val, 0, percentile, ls="--", lw=1, colors="k")
        ax.text(val + 0.1, 0, f"{val:0.1f}m/s", ha="left", va="bottom")
        ax.text(
            0.1,
            percentile + 0.02 if percentile < 0.1 else percentile - 0.02,
            f"{percentile:0.0%}",
            ha="left",
            va="bottom" if percentile < 0.1 else "top",
        )

    ax.set_ylim(0, 1.01)

    if title is not None:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return fig


def seasonal_speed(
    wind_speeds: pd.Series,
    percentiles: Tuple[float] = (0.1, 0.25, 0.75, 0.9),  # type: ignore
    color: str = "k",
    title: str = "",
) -> plt.Figure:  # type: ignore
    """Plot the wind-speed/frequency histogram for collection of wind speeds.

    Args:
        wind_speeds (pd.Series):
            A time-indexed collection of wind speeds.
        percentiles (Tuple[float], optional):
            A list of percentiles to show on the chart. Defaults to (0.1, 0.25, 0.75, 0.9).
        color (str, optional):
            The color of the plot. Defaults to "k".
        title (str, optional):
            A title to add to the resulting plot. Defaults to None.

    Returns:
        plt.Figure:
            A Figure object.
    """
    # if speed_bins is None:
    #     speed_bins = np.linspace(min(wind_speeds), np.quantile(wind_speeds, 0.999), 16)

    # if percentiles and ((min(percentiles) < 0) or (max(percentiles) > 1)):
    #     raise ValueError("percentiles must fall within the range 0-1.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.hist(wind_speeds, bins=speed_bins, density=True, color="grey")

    # if weibull != False:
    #     if isinstance(weibull, bool):
    #         params = weibull_pdf(wind_speeds)
    #     elif not isinstance(weibull, bool):
    #         params = weibull
    #     new_x = np.linspace(min(speed_bins), max(speed_bins), 100)
    #     new_y = exponweib.pdf(new_x, *params)
    #     ax.plot(new_x, new_y, label="Weibull (PDF)", c="k")

    # low, _ = ax.get_ylim()
    # for percentile in percentiles:
    #     x = np.quantile(wind_speeds, percentile)
    #     ax.axvline(x, 0, 1, ls="--", lw=1, c="k")
    #     ax.text(x - 0.1, low, f"{percentile:0.0%}", ha="right", va="bottom")
    #     ax.text(x + 0.1, low, f"{x:0.1f}m/s", ha="left", va="bottom")

    # ax.set_xlim(0, max(speed_bins))
    # ax.set_xlabel("Wind Speed (m/s)")
    # ax.set_ylabel("Frequency")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))

    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)
    # ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    # ax.legend(
    #     bbox_to_anchor=(1, 1),
    #     ncol=1,
    #     loc="upper right",
    #     borderaxespad=0.0,
    #     frameon=False,
    #     fontsize="small",
    # )

    # if title is not None:
    #     ax.set_title(title, x=0, ha="left", va="bottom")

    # plt.tight_layout()

    return fig


def speed_frequency(
    wind_speeds: List[float],
    speed_bins: List[float] = None,
    weibull: Union[bool, Tuple[float]] = False,
    percentiles: Tuple[float] = (),
    title: str = None,
) -> plt.Figure:
    """Plot the wind-speed/frequency histogram for collection of wind speeds.

    Args:
        wind_speeds (List[float]):
            A collection of wind speeds.
        speed_bins (List[float], optional):
            A set of bins to fit the input wind speeds into. Defaults to None.
        weibull (Union[bool, Tuple[float]], optional):
            Include the weibull curve on the plot. Defaults to False.
        percentiles (Tuple[float], optional):
            A list of percentiles to show on the chart. Defaults to None.
        title (str, optional):
            A title to add to the resulting plot. Defaults to None.

    Returns:
        plt.Figure:
            A Figure object.
    """
    if speed_bins is None:
        speed_bins = np.linspace(min(wind_speeds), np.quantile(wind_speeds, 0.999), 16)

    if percentiles and ((min(percentiles) < 0) or (max(percentiles) > 1)):
        raise ValueError("percentiles must fall within the range 0-1.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(wind_speeds, bins=speed_bins, density=True, color="grey")

    if weibull != False:
        if isinstance(weibull, bool):
            params = weibull_pdf(wind_speeds)
        elif not isinstance(weibull, bool):
            params = weibull
        new_x = np.linspace(min(speed_bins), max(speed_bins), 100)
        new_y = exponweib.pdf(new_x, *params)
        ax.plot(new_x, new_y, label="Weibull (PDF)", c="k")

    low, _ = ax.get_ylim()
    for percentile in percentiles:
        x = np.quantile(wind_speeds, percentile)
        ax.axvline(x, 0, 1, ls="--", lw=1, c="k")
        ax.text(x - 0.1, low, f"{percentile:0.0%}", ha="right", va="bottom")
        ax.text(x + 0.1, low, f"{x:0.1f}m/s", ha="left", va="bottom")

    ax.set_xlim(0, max(speed_bins))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    ax.legend(
        bbox_to_anchor=(1, 1),
        ncol=1,
        loc="upper right",
        borderaxespad=0.0,
        frameon=False,
        fontsize="small",
    )

    if title is not None:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return fig


def timeseries(
    wind_speeds: pd.Series, color: str = "k", title: str = None
) -> plt.Figure:
    """Plot a time-series of wind speeds

    Args:
        wind_speeds (pd.Series): _description_
        color (str, optional): _description_. Defaults to "k".
        title (str, optional): _description_. Defaults to None.

    Returns:
        plt.Figure: _description_
    """

    if not isinstance(wind_speeds.index, pd.DatetimeIndex):
        raise ValueError("The wind_speeds given should be datetime-indexed.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    wind_speeds.plot(ax=ax, c=color, lw=0.5)

    ax.set_xlim(wind_speeds.index.min(), wind_speeds.index.max())
    ax.set_ylabel("Wind Speed (m/s)")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.25)

    if title is not None:
        ax.set_title(title, x=0, color="k", ha="left")

    plt.tight_layout()

    return fig


def windrose(
    wind_directions: List[float],
    data: List[float] = None,
    direction_bins: DirectionBins = DirectionBins(),
    data_bins: List[float] = None,
    cmap: Union[Colormap, str] = None,
    title: str = None,
    include_legend: bool = True,
    include_percentages: bool = False,
) -> plt.Figure:
    """Plot a windrose for a collection of wind speeds and directions.

    Args:
        wind_directions (List[float]):
            A collection of wind-directions.
        data (List[float]):
            A collection of direction-associated data.
        direction_bins (DirectionBins, optional):
            A DirectionBins object.
        data_bins (List[float], optional):
            Bins to sort data into. Defaults to the boundaries for Beaufort wind conditions.
        cmap (Union[Colormap, str], optional):
            Use a custom colormap. Defaults to "GnBu_r".
        title (str, optional):
            Add a title to the plot. Defaults to None.
        include_legend (bool, optional):
            Set to True to include the legend. Defaults to True.
        include_percentages (bool, optional):
            Add bin totals as % to rose. Defaults to False.

    Returns:
        plt.Figure:
            A Figure object.
    """

    # set data binning defaults (beaufort bins)
    if data_bins is None:
        data_bins = [
            0,
            0.3,
            1.5,
            3.3,
            5.5,
            7.9,
            10.7,
            13.8,
            17.1,
            20.7,
            24.4,
            28.4,
            32.6,
        ]

    # set cmap defaults
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if cmap is None:
        cmap = ListedColormap(
            colors=[
                "#FFFFFF",
                "#CCFFFF",
                "#99FFCC",
                "#99FF99",
                "#99FF66",
                "#99FF00",
                "#CCFF00",
                "#FFFF00",
                "#FFCC00",
                "#FF9900",
                "#FF6600",
                "#FF3300",
                "#FF0000",
            ]
        )

    # bin input data
    thetas = np.deg2rad(direction_bins.midpoints)
    width = np.deg2rad(direction_bins.bin_width)
    binned_data = direction_bins.bin_data(wind_directions, data)
    radiis = np.array(
        [
            np.histogram(values, data_bins, density=False)[0]
            for _, values in binned_data.items()
        ]
    )
    bottoms = np.vstack([[0] * len(direction_bins.midpoints), radiis.cumsum(axis=1).T])[
        :-1
    ].T
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_bins) - 1)]

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # plot binned data
    for theta, radii, bottom in zip(*[thetas, radiis, bottoms]):
        bars = ax.bar(theta, radii, width=width, bottom=bottom, color=colors, zorder=2)

    if include_percentages:
        percentages = [
            len(vals) / len(wind_directions)
            for (low, high), vals in binned_data.items()
        ]
        for theta, radii, percentage in zip(*[thetas, radiis.sum(axis=1), percentages]):
            tt = ax.text(
                theta,
                radii,
                f"{percentage:0.1%}",
                fontsize="x-small",
                ha="center",
                va="center",
            )
            tt.set_bbox(dict(facecolor="white", alpha=0.5, linewidth=0))

    # format plot area
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.get_yticklabels(), fontsize="small")
    ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
    ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
    ax.set_xticks(
        np.radians(
            (22.5, 45, 67.5, 112.5, 135, 157.5, 202.5, 225, 247.5, 292.5, 315, 337.5)
        ),
        minor=True,
    )
    ax.set_xticklabels(
        (
            "NNE",
            "NE",
            "ENE",
            "ESE",
            "SE",
            "SSE",
            "SSW",
            "SW",
            "WSW",
            "WNW",
            "NW",
            "NNW",
        ),
        minor=True,
        **{"fontsize": "x-small"},
    )

    # construct legend
    if include_legend:
        handles = [
            mpatches.Patch(color=colors[n], label=f"{i} to {j}")
            for n, (i, j) in enumerate(rolling_window(data_bins, 2))
        ]
        lgd = ax.legend(
            handles=handles,
            bbox_to_anchor=(1.1, 0.5),
            loc="center left",
            ncol=1,
            borderaxespad=0,
            frameon=False,
            fontsize="small",
            title="m/s",
        )

    if title:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return fig
