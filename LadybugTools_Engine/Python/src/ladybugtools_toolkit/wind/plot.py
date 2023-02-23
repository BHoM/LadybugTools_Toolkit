from __future__ import annotations

from calendar import month_abbr
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
    wind_direction: List[float],
    data: List[float] = None,
    direction_bins: DirectionBins = DirectionBins(),
    data_bins: Union[int, List[float]] = None,
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
    if isinstance(data_bins, int):
        data_bins = np.linspace(min(data), max(data), data_bins + 1).round(1)

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
    binned_data = direction_bins.bin_data(wind_direction, data)
    radiis = np.array(
        [
            np.histogram(a=values, bins=data_bins, density=False)[0]
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
        _ = ax.bar(theta, radii, width=width, bottom=bottom, color=colors, zorder=2)

    if include_percentages:
        percentages = [
            len(vals) / len(wind_direction) for (low, high), vals in binned_data.items()
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
        _ = ax.legend(
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


def windhist(
    wind_direction_bins: List[float],
    radial_bins: List[float],
    values: List[List[float]],
    cmap: Union[Colormap, str] = None,
    include_labels: bool = False,
    include_cbar: bool = True,
    cmap_label: str = None,
    cbar_freq: bool = False,
    title: str = None,
) -> plt.Figure:
    """Create a radial 2d heatmap-histogram.

    Args:
        wind_direction_bins (List[float]): _description_
        radial_bins (List[float]): _description_
        values (List[List[float]]): _description_
        cmap (Union[Colormap, str], optional): _description_. Defaults to None.
        include_labels (bool, optional): _description_. Defaults to False.
        include_cbar (bool, optional): _description_. Defaults to True.
        cmap_label (str, optional): _description_. Defaults to None.
        cbar_freq (bool, optional): _description_. Defaults to False.
        title (str, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        plt.Figure: _description_
    """

    # set cmap defaults
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if cmap is None:
        cmap = plt.get_cmap("magma_r")

    # plot figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    pc = ax.pcolormesh(
        wind_direction_bins,
        radial_bins,
        values,
        cmap=cmap,
        alpha=1,
        ec="none",
        lw=0,
        zorder=0,
    )

    # format plot area
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.25, c="k")
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

    if include_cbar:
        cb = fig.colorbar(
            pc, pad=0.07, drawedges=False, label=cmap_label, extend="max", aspect=50
        )
        cb.outline.set_visible(False)
        if cbar_freq:
            cb.ax.axes.yaxis.set_major_formatter(mtick.PercentFormatter(1))

    if include_labels:
        raise NotImplementedError("TODO - add functionality")

    if title:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()


def windrose_matrix(
    wind_direction: pd.Series,
    data: pd.Series = None,
    month_bins: Tuple[List[int]] = ([12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]),
    hour_bins: Tuple[List[int]] = (
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
    ),
    direction_bins: DirectionBins = DirectionBins(),
    data_bins: Union[int, List[float]] = None,
    cmap: Union[Colormap, str] = None,
    title: str = None,
) -> plt.Figure:
    if len(wind_direction) != len(data):
        raise ValueError("Input directions and data are not the same length.")

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
    if isinstance(data_bins, int):
        data_bins = np.linspace(min(data), max(data), data_bins + 1).round(1)

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

    # determine how many rows and columns
    n_rows = len(month_bins)
    n_cols = len(hour_bins)

    # create col/row labels
    col_labels = [", ".join([month_abbr[j] for j in i]) for i in month_bins]
    row_labels = [f"{i[0]:02d}:00 to {i[-1]:02d}:00" for i in hour_bins]

    # create colors to apply to each bin
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_bins) - 1)]

    # get bin sizes
    thetas = np.deg2rad(direction_bins.midpoints)
    width = np.deg2rad(direction_bins.bin_width)

    # create plot opbject to populate
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_rows, 2 * n_cols),
        subplot_kw={"projection": "polar"},
    )
    [ax.set_theta_zero_location("N") for ax in axes.flatten()]
    [ax.set_theta_direction(-1) for ax in axes.flatten()]

    # for each row and column, filter the input data by hours/months and generate windrose
    for n, month_bin in enumerate(month_bins):
        # add column labels here
        for m, hour_bin in enumerate(hour_bins):
            # add row labels here (if it's the first one)
            # if m == 0:
            #     axes[n][m].text(0, 0.5, row_labels[n], transform=axes[n][m].transaxes())
            mask = wind_direction.index.month.isin(
                month_bin
            ) & wind_direction.index.hour.isin(hour_bin)
            # bin input data
            binned_data = direction_bins.bin_data(wind_direction[mask], data[mask])
            radiis = np.array(
                [
                    np.histogram(a=values, bins=data_bins, density=False)[0]
                    for _, values in binned_data.items()
                ]
            )
            bottoms = np.vstack(
                [[0] * len(direction_bins.midpoints), radiis.cumsum(axis=1).T]
            )[:-1].T

            # plot binned data
            for theta, radii, bottom in zip(*[thetas, radiis, bottoms]):
                _ = axes[n][m].bar(
                    theta, radii, width=width, bottom=bottom, color=colors, zorder=2
                )
            axes[n][m].set_title(
                f"{col_labels[n]}\n{row_labels[m]}",
                fontsize="xx-small",
                ha="left",
                va="bottom",
            )

    # format plot area
    for ax in axes.flatten():
        ax.spines["polar"].set_visible(False)
        ax.grid(True, which="both", ls="--", zorder=0, alpha=0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.get_yticklabels(), fontsize="xx-small")
        ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
        ax.set_xticklabels(
            ("N", "E", "S", "W"), minor=False, **{"fontsize": "xx-small"}
        )

    # -- Creating a new axes at the right side
    cbax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cbax.set_aspect("equal")
    cbax.axis("off")
    # -- Plotting the colormap in the created axes
    handles = [
        mpatches.Patch(color=colors[n], label=f"{i} to {j}")
        for n, (i, j) in enumerate(rolling_window(data_bins, 2))
    ]
    _ = cbax.legend(
        handles=handles,
        bbox_to_anchor=(1.1, 0.5),
        loc="center left",
        ncol=1,
        borderaxespad=0,
        frameon=False,
        fontsize="xx-small",
        title="m/s",
    )
    fig.subplots_adjust(left=0.05, right=0.85)

    if title is not None:
        plt.suptitle(title, x=0, ha="left", va="bottom")

    return fig
