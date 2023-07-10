import calendar
import warnings
from typing import List, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from scipy import stats

from ..categorical.categories import CategoriesBase
from ..helpers import (
    angle_from_north,
    cardinality,
    contrasting_color,
    rolling_window,
    wind_direction_average,
)
from ..wind.direction_bins import DirectionBins
from .utilities import annotate_imshow

# TODO - add Climate Consultant-style "wind wheel" plot here (http://2.bp.blogspot.com/-F27rpZL4VSs/VngYxXsYaTI/AAAAAAAACAc/yoGXmk13uf8/s1600/CC-graphics%2B-%2BWind%2BWheel.jpg)


def windrose(
    wind_directions: List[float],
    data: List[float] = None,
    ax: plt.Axes = None,
    direction_bins: DirectionBins = DirectionBins(),
    data_bins: Union[int, List[float], CategoriesBase] = 11,
    include_legend: bool = True,
    include_percentages: bool = False,
    **kwargs,
) -> plt.Axes:
    """Plot a windrose for a collection of wind speeds and directions.

    Args:
        wind_directions (List[float]):
            A collection of wind-directions.
        data (List[float]):
            A collection of direction-associated data.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        direction_bins (DirectionBins, optional):
            A DirectionBins object.
        data_bins (List[float], optional):
            Bins to sort data into. Defaults to the boundaries for Beaufort wind conditions.
        include_legend (bool, optional):
            Set to True to include the legend. Defaults to True.
        include_percentages (bool, optional):
            Add bin totals as % to rose. Defaults to False.
        **kwargs:
            Additional keyword arguments to pass to the function.
            data_unit (str, optional):
                The unit of the data to add to the legend. Defaults to None.
            ylim (Tuple[float], optional):
                The minimum and maximum values for the y-axis. Defaults to None.
            cmap (str, optional):
                The name of the colormap to use. Defaults to "viridis". If data_bins is CategoriesBase, then this will override the CategoriesBase.cmap() value.

    Returns:
        plt.Figure:
            A Figure object.
    """

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    title = [
        kwargs.pop("title", None),
    ]
    ax.set_title("\n".join([i for i in title if i is not None]))

    # set data binning defaults
    _data_bins: List[float] = data_bins
    if isinstance(data_bins, int):
        _data_bins = np.linspace(min(data), max(data), data_bins + 1).round(1)
    elif isinstance(data_bins, CategoriesBase):
        _data_bins = data_bins._bin_edges()

    # get colormap
    cmap = plt.get_cmap(
        kwargs.pop(
            "cmap",
            "viridis" if not isinstance(data_bins, CategoriesBase) else data_bins.cmap,
        )
    )

    # bin input data
    thetas = np.deg2rad(direction_bins.midpoints)
    width = np.deg2rad(direction_bins.bin_width)
    binned_data = direction_bins.bin_data(wind_directions, data)
    radiis = np.array(
        [
            np.nan_to_num(np.histogram(a=values, bins=_data_bins)[0])
            for _, values in binned_data.items()
        ]
    )
    bottoms = np.vstack([[0] * len(direction_bins.midpoints), radiis.cumsum(axis=1).T])[
        :-1
    ].T
    colors = [cmap(i) for i in np.linspace(0, 1, len(_data_bins) - 1)]

    # create figure
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # plot binned data
    for theta, radii, bottom in zip(*[thetas, radiis, bottoms]):
        _ = ax.bar(theta, radii, width=width, bottom=bottom, color=colors, zorder=2)

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

    ax.set_ylim(kwargs.pop("ylim", None))

    # construct legend
    if include_legend:
        handles = [
            mpatches.Patch(color=colors[n], label=f"{i} to {j}")
            for n, (i, j) in enumerate(rolling_window(_data_bins, 2))
        ]
        _ = ax.legend(
            handles=handles,
            bbox_to_anchor=(1.1, 0.5),
            loc="center left",
            ncol=1,
            borderaxespad=0,
            frameon=False,
            fontsize="small",
            title=kwargs.pop("data_unit", None),
        )

    plt.tight_layout()

    return ax


def wind_matrix(
    wind_speeds: pd.Series,
    wind_directions: pd.Series,
    ax: plt.Axes = None,
    additional_data: pd.Series = None,
    show_values: bool = False,
    **kwargs,
) -> plt.Axes:
    """Generate a wind-speed/direction matrix from a set of time-indexed speed and direction values.

    Args:
        wind_speeds (pd.Series):
            A list of time-indexed wind speeds.
        wind_directions (pd.Series):
            A list of time-indexed wind directions (in degrees from north clockwise).
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        additional_data (pd.Series, optional):
            A list of time-indexed additional data values to plot as a heatmap, aligning with the wind speeds and directions. Defaults to None.
        show_values (bool, optional):
            Show the values in the matrix. Defaults to False.
        data_lims (Tuple[float], optional):
            The minimum and maximum data values to use for the colorbar. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib pcolor function.
            vmin (float, optional):
                The minimum value to use for the colorbar. Defaults to None.
            vmax (float, optional):
                The maximum value to use for the colorbar. Defaults to None.
            cmap (str, optional):
                The name of the colormap to use. Defaults to "Spectral_r".
            cbar_title (str, optional):
                The title to use for the colorbar. Defaults to None.
            edgecolor (str, optional):
                The color to use for the grid lines. Defaults to None.
    Returns:
        Figure:
            A matplotlib figure object.
    """

    if ax is None:
        ax = plt.gca()

    title = [
        kwargs.pop("title", None),
        "Wind speed/direction matrix",
    ]
    ax.set_title("\n".join([i for i in title if i is not None]))

    # convert time-indexed data to 24*12 matrix
    _wind_speeds = (
        wind_speeds.groupby([wind_speeds.index.hour, wind_speeds.index.month], axis=0)
        .mean()
        .unstack()
    )
    _wind_directions = (
        (
            (
                wind_directions.groupby(
                    [wind_directions.index.month, wind_directions.index.hour], axis=0
                ).apply(wind_direction_average)
                # + 90
            )
            % 360
        )
        .unstack()
        .T
    )

    if any(
        [
            _wind_speeds.shape != (24, 12),
            _wind_directions.shape != (24, 12),
            _wind_directions.shape != _wind_speeds.shape,
            not np.array_equal(_wind_directions.index, _wind_speeds.index),
            not np.array_equal(_wind_directions.columns, _wind_speeds.columns),
        ]
    ):
        raise ValueError(
            "The wind_speeds and wind_directions must cover all months of the year, and all hours of the day, and align with each other."
        )

    if additional_data is None:
        _additional_data = _wind_speeds
    else:
        _additional_data = (
            additional_data.groupby(
                [additional_data.index.hour, additional_data.index.month], axis=0
            )
            .mean()
            .unstack()
        )
    if any(
        [
            _additional_data.shape != _wind_speeds.shape,
            not np.array_equal(_additional_data.index, _wind_speeds.index),
            not np.array_equal(_additional_data.columns, _wind_speeds.columns),
        ]
    ):
        raise ValueError(
            "The additional_data must align with the wind_speeds and wind_directions."
        )

    # get kwargs
    if (kwargs.get("norm", None) is None) and additional_data is not None:
        warnings.warn(
            "vmin and vmax should be set when using a normalised colorbar, otherwise color mapping may look odd."
        )

    cmap = kwargs.pop("cmap", "Spectral_r")
    vmin = kwargs.pop("vmin", _additional_data.values.min())
    vmax = kwargs.pop("vmax", _additional_data.values.max())
    cbar_title = kwargs.pop("cbar_title", None)
    norm = kwargs.pop("norm", Normalize(vmin=vmin, vmax=vmax, clip=True))
    mapper = kwargs.pop("mapper", ScalarMappable(norm=norm, cmap=cmap))

    pc = ax.pcolor(_additional_data, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    _x = -np.sin(np.deg2rad(_wind_directions.values))
    _y = -np.cos(np.deg2rad(_wind_directions.values))
    direction_matrix = angle_from_north([_x, _y])
    ax.quiver(
        np.arange(1, 13, 1) - 0.5,
        np.arange(0, 24, 1) + 0.5,
        _x * _wind_speeds.values / 2,
        _y * _wind_speeds.values / 2,
        pivot="mid",
        fc="white",
        ec="black",
        lw=0.5,
        alpha=0.5,
    )

    if show_values:
        for _xx, col in enumerate(_wind_directions.values.T):
            for _yy, _ in enumerate(col.T):
                local_value = _additional_data.values[_yy, _xx]
                # print(local_value)
                cell_color = mapper.to_rgba(local_value)
                text_color = contrasting_color(cell_color)
                # direction text
                ax.text(
                    _xx,
                    _yy,
                    f"{direction_matrix[_yy][_xx]:0.0f}°",
                    color=text_color,
                    ha="left",
                    va="bottom",
                    fontsize="xx-small",
                )
                # speed text
                ax.text(
                    _xx + 1,
                    _yy + 1,
                    f"{_wind_speeds.values[_yy][_xx]:0.1f}m/s",
                    color=text_color,
                    ha="right",
                    va="top",
                    fontsize="xx-small",
                )
    ax.set_xticks(np.arange(1, 13, 1) - 0.5)
    ax.set_xticklabels([calendar.month_abbr[i] for i in np.arange(1, 13, 1)])
    ax.set_yticks(np.arange(0, 24, 1) + 0.5)
    ax.set_yticklabels([f"{i:02d}:00" for i in np.arange(0, 24, 1)])
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    cb = plt.colorbar(pc, label=cbar_title, pad=0.01)
    cb.outline.set_visible(False)

    return ax


def wind_speed_direction_frequency(
    wind_speed: List[float],
    wind_direction: List[float],
    ax: plt.Axes = None,
    speed_bins: List[float] = None,
    n_directions: float = 8,
    **kwargs,
) -> plt.Axes:
    """Create an image containing a matrix of wind speed/direction frequencies.

    Args:
        wind_speed (List[float]):
            A list if wind speeds to process.
        wind_direction (List[float]):
            A list of wind directions to process.
        ax (plt.Axes, optional):
            The matplotlib Axes to plot on. Defaults to None which uses the current Axes.
        speed_bins (List[float], optional):
            A set of bins into which wind speed will be sorted. Defaults to None which divides the
            range 0-MAX into 10 bins.
        n_directions (float, optional):
            A number of cardinal directions into which wind_direction will be grouped.
            Default is 8.
        **kwargs:
            Additional keyword arguments to pass to the function.
    Returns:
        plt.Axes:
            A matplotlib Axes object.
    """

    cmap = kwargs.pop("cmap", "Purples")

    if ax is None:
        ax = plt.gca()

    if len(wind_speed) != len(wind_direction):
        raise ValueError("The given wind speeds and directions do not align.")

    if speed_bins is None:
        speed_bins = np.linspace(0, max(wind_speed), 11)

    # create "bins" of direction based on number of directions to bin into
    direction_bin_width = 360 / n_directions
    direction_bins = (
        [0]
        + (np.linspace(0, 360, n_directions + 1) + (direction_bin_width / 2)).tolist()[
            :-1
        ]
        + [360]
    )

    # bin data into table
    binned, speed_bins, direction_bins = np.histogram2d(
        wind_speed, wind_direction, bins=[speed_bins, direction_bins], normed=False
    )
    df = pd.DataFrame(binned)
    # combine first and last column, and drop last column (to get North winds)
    df.iloc[:, 0] = df.iloc[:, 0] + df.iloc[:, -1]
    df = df.iloc[:, :-1]

    speed_index = [(i[0], i[1]) for i in rolling_window(speed_bins, 2)]
    df.index = speed_index

    if n_directions in [4, 8, 16, 32]:
        direction_index = [
            cardinality(i, n_directions)
            for i in [0] + rolling_window(direction_bins[1:-1], 2).mean(axis=1).tolist()
        ]
    else:
        direction_index = [
            (i[0], i[1])
            for i in np.roll(
                rolling_window(direction_bins[1:-1].tolist() + [direction_bins[1]], 2),
                1,
                axis=0,
            )
        ]
    df.columns = direction_index
    df = df / len(wind_speed)

    ax.set_aspect("auto")
    im = ax.imshow(df.values, cmap=cmap)

    annotate_imshow(im, valfmt="{x:.02%}", fontsize="x-small", exclude_vals=[0])

    ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(df.columns))))

    if isinstance(df.columns[0], str):
        xticklabels = df.columns
    else:
        xticklabels = [f"{i[0]:0.1f}-\n{i[1]:0.1f}" for i in df.columns]
    ax.set_xticklabels(
        xticklabels,
        rotation=0,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("Wind direction (degrees)")

    ax.yaxis.set_major_locator(mticker.FixedLocator(range(len(df.index))))
    yticklabels = [f"{i[0]}-{i[1]}" for i in df.index]
    ax.set_yticklabels(yticklabels, rotation=0, ha="right", va="center", fontsize=8)
    ax.set_ylabel("Wind speed (m/s)")

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

    ti = "Wind speed/direction frequency"
    ax.set_title(ti, x=0, ha="left")

    plt.tight_layout()

    return ax


def wind_cumulative_probability(
    wind_speeds: Tuple[float],
    ax: plt.Axes = None,
    speed_bins: Tuple[float] = None,
    percentiles: Tuple[float] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a cumulative probability graph, showing binned wind speeds in a cumulative
        frequency histogram.

    Args:
        wind_speeds (List[float]):
            A list of wind speeds.
        ax (plt.Axes, optional):
            A matplotlib Axes object. Defaults to None.
        speed_bins (List[float], optional):
            A set of bin edges to categorise the wind speeds into. Defaults to None.
        percentiles (List[float], optional):
            A list of percentiles to show on the chart. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the plotting function.

    Returns:
        plt.Figure:
            A Figure object.
    """
    if ax is None:
        ax = plt.gca()

    if percentiles is None:
        percentiles = [0.5, 0.95]
    if (min(percentiles) < 0) or (max(percentiles) > 1):
        raise ValueError("percentiles must fall within the range 0-1.")

    if speed_bins is None:
        speed_bins = np.linspace(0, 25, 50)
    if min(speed_bins) < 0:
        raise ValueError("Minimum bin value must be >= 0")

    x = speed_bins
    y = [stats.percentileofscore(wind_speeds, i) / 100 for i in speed_bins]

    # remove all but one element from lists where probability is == 1
    if len([i for i in y if i == 1]) > 2:
        idxmax = np.where([i == 1 for i in y])[0][0]
        x = x[: idxmax + 1]
        y = y[: idxmax + 1]

    ax.plot(x, y, c="grey")
    ax.set_xlim(0, max(x))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=0))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.25)

    for percentile in percentiles:
        val = np.quantile(wind_speeds, percentile)
        ax.hlines(percentile, 0, val, ls="--", lw=1, colors="black")
        ax.vlines(val, 0, percentile, ls="--", lw=1, colors="black")
        ax.text(val + 0.1, 0, f"{val:0.1f}m/s", ha="left", va="bottom")
        ax.text(
            0.1,
            percentile + 0.02 if percentile < 0.1 else percentile - 0.02,
            f"{percentile:0.0%}",
            ha="left",
            va="bottom" if percentile < 0.1 else "top",
        )

    ax.set_ylim(0, 1.01)

    plt.tight_layout()

    return ax


def seasonal_speed(
    wind_speeds: pd.Series,
    ax: plt.Axes = None,
    percentiles: Tuple[float] = (0.1, 0.25, 0.75, 0.9),  # type: ignore
    color: str = "black",
    title: str = "",
    **kwargs,
) -> plt.Axes:  # type: ignore
    """Plot the wind-speed/frequency histogram for collection of wind speeds.

    Args:
        wind_speeds (pd.Series):
            A time-indexed collection of wind speeds.
        percentiles (Tuple[float], optional):
            A list of percentiles to show on the chart. Defaults to (0.1, 0.25, 0.75, 0.9).
        color (str, optional):
            The color of the plot. Defaults to "black".
        title (str, optional):
            A title to add to the resulting plot. Defaults to None.

    Returns:
        plt.Figure:
            A Figure object.
    """

    # if ax is None:
    #     ax = plt.gca()

    # if speed_bins is None:
    #     speed_bins = np.linspace(min(wind_speeds), np.quantile(wind_speeds, 0.999), 16)

    # if percentiles and ((min(percentiles) < 0) or (max(percentiles) > 1)):
    #     raise ValueError("percentiles must fall within the range 0-1.")

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

    # TODO - reimplement figure and kwargs usage.
    raise NotImplementedError("This function is not fully implemented.")


def wind_speed_frequency(
    wind_speeds: List[float],
    ax: plt.Axes = None,
    speed_bins: List[float] = None,
    # weibull: Union[bool, Tuple[float]] = False,
    percentiles: Tuple[float] = (),
    **kwargs,
) -> plt.Axes:
    """Plot the wind-speed/frequency histogram for collection of wind speeds.

    Args:
        wind_speeds (List[float]):
            A collection of wind speeds.
        ax (plt.Axes, optional):
            A matplotlib Axes object to plot on. Defaults to None.
        speed_bins (List[float], optional):
            A set of bins to fit the input wind speeds into. Defaults to None.
        weibull (Union[bool, Tuple[float]], optional):
            Include the weibull curve on the plot. Defaults to False.
        percentiles (Tuple[float], optional):
            A list of percentiles to show on the chart. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the matplotlib hist function.

    Returns:
        plt.Figure:
            A Figure object.
    """

    if ax is None:
        ax = plt.gca()

    if speed_bins is None:
        speed_bins = np.linspace(min(wind_speeds), np.quantile(wind_speeds, 0.999), 16)

    if percentiles and ((min(percentiles) < 0) or (max(percentiles) > 1)):
        raise ValueError("percentiles must fall within the range 0-1.")

    ax.hist(wind_speeds, bins=speed_bins, density=True, color="grey")

    # if weibull != False:
    #     if isinstance(weibull, bool):
    #         params = weibull_pdf(wind_speeds)
    #     elif not isinstance(weibull, bool):
    #         params = weibull
    #     new_x = np.linspace(min(speed_bins), max(speed_bins), 100)
    #     new_y = weibull_min.pdf(new_x, *params)
    #     ax.plot(new_x, new_y, label="Weibull (PDF)", c="k")

    low, _ = ax.get_ylim()
    for percentile in percentiles:
        x = np.quantile(wind_speeds, percentile)
        ax.axvline(x, 0, 1, ls="--", lw=1, c="black")
        ax.text(x - 0.1, low, f"{percentile:0.0%}", ha="right", va="bottom")
        ax.text(x + 0.1, low, f"{x:0.1f}m/s", ha="left", va="bottom")

    ax.set_xlim(0, max(speed_bins))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.25)

    ax.legend(
        bbox_to_anchor=(1, 1),
        ncol=1,
        loc="upper right",
        borderaxespad=0.0,
        frameon=False,
        fontsize="small",
    )

    plt.tight_layout()

    return ax


def radial_histogram(
    radial_values: List[float],
    radial_bins: List[float],
    values: List[List[float]],
    ax: plt.Axes = None,
    cmap: Union[Colormap, str] = None,
    include_labels: bool = False,
    include_cbar: bool = True,
    cmap_label: str = None,
    cbar_freq: bool = False,
    title: str = None,
    **kwargs,
) -> plt.Axes:
    """Create a radial 2d heatmap-histogram.

    Args:
        radial_values (List[float]): _description_
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
        plt.Axes: _description_
    """

    warnings.warn(f"This method is not fully formed!")

    # TODO - deal with kwargs

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # set cmap defaults
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if cmap is None:
        cmap = plt.get_cmap("magma_r")

    # plot figure
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    pc = ax.pcolormesh(
        radial_values,
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
    ax.grid(True, which="both", ls="--", zorder=0, alpha=0.25)
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
        cb = plt.colorbar(
            pc,
            ax=ax,
            pad=0.07,
            drawedges=False,
            label=cmap_label,
            extend="max",
            aspect=50,
        )
        cb.outline.set_visible(False)
        if cbar_freq:
            cb.ax.axes.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    if include_labels:
        raise NotImplementedError("TODO - add functionality")

    if title:
        ax.set_title(title, x=0, ha="left", va="bottom")

    plt.tight_layout()

    return ax
