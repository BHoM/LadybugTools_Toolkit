from __future__ import annotations

import colorsys
import copy
from pkgutil import extend_path
import textwrap
from datetime import datetime
from typing import List, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug_extension.datacollection import to_series
from matplotlib import patches
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    cnames,
    is_color_like,
    rgb2hex,
    to_rgb,
)
from matplotlib.figure import Figure
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def colormap_sequential(colors: List[Union[str, Tuple]]) -> LinearSegmentedColormap:
    """Create a sequential colormap from a list of input colors.
    Args:
        colors (List[Union[str, Tuple]]): A list of colors according to their hex-code, string name, character code or RGBA values.
    Raises:
        KeyError: Where a given color string is not recognized, an error will be raised.
    Returns:
        LinearSegmentedColormap: A matplotlib colormap.
    """

    assert (
        type(colors) is list and len(colors) >= 2
    ), "The input is not a list of named colors, or is only one color long."

    fixed_colors = []
    for c in colors:
        if is_color_like(c):
            try:
                fixed_colors.append(rgb2hex(c))
            except:
                fixed_colors.append(c)
        else:
            raise KeyError(f"{c} not recognised as a valid color string.")
    return LinearSegmentedColormap.from_list(
        f"{'_'.join(fixed_colors)}",
        list(zip(np.linspace(0, 1, len(fixed_colors)), fixed_colors)),
        N=256,
    )


UTCI_COLORMAP = ListedColormap(
    [
        "#262972",
        "#3452A4",
        "#3C65AF",
        "#37BCED",
        "#2EB349",
        "#F38322",
        "#C31F25",
        "#7F1416",
    ]
)
UTCI_COLORMAP.set_under("#0D104B")
UTCI_COLORMAP.set_over("#580002")
UTCI_LEVELS = [-40, -27, -13, 0, 9, 26, 32, 38, 46]
UTCI_LABELS = [
    "Extreme Cold Stress",
    "Very Strong Cold Stress",
    "Strong Cold Stress",
    "Moderate Cold Stress",
    "Slight Cold Stress",
    "No Thermal Stress",
    "Moderate Heat Stress",
    "Strong Heat Stress",
    "Very Strong Heat Stress",
    "Extreme Heat Stress",
]
UTCI_BOUNDARYNORM = BoundaryNorm(UTCI_LEVELS, UTCI_COLORMAP.N)

DBT_COLORMAP = colormap_sequential(["white", "#bc204b"])
RH_COLORMAP = colormap_sequential(["white", "#8db9ca"])
MRT_COLORMAP = colormap_sequential(["white", "#6d104e"])
WS_COLORMAP = colormap_sequential(
    [
        "#d0e8e4",
        "#8db9ca",
        "#006da8",
        "#24135f",
    ]
)


def create_triangulation(
    x: List[float], y: List[float], alpha: float = 1.1
) -> Triangulation:
    """Create a matplotlib Triangulation from a list of x and y coordinates, including a mask to remove islands within this triangulation.

    Args:
        x (List[float]): A list of x coordinates.
        y (List[float]): A list of y coordinates.
        alpha (float, optional): A value to start alpha at. Defaults to 1.1.

    Returns:
        Triangulation: A matploltib Triangulation object.
    """

    if len(x) != len(y):
        raise ValueError("x and y must be the same length")

    # Traingulate X, Y locations
    triang = Triangulation(x, y)

    xtri = x[triang.triangles] - np.roll(x[triang.triangles], 1, axis=1)
    ytri = y[triang.triangles] - np.roll(y[triang.triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)

    # Iterate triangulation masking until a possible mask is found
    count = 0
    max_iterations = 250
    alpha = alpha
    increment = 0.01
    fig, ax = plt.subplots(1, 1)
    synthetic_values = range(len(x))
    success = False
    while not success:
        count += 1
        try:
            tr = copy.deepcopy(triang)
            tr.set_mask(maxi > alpha)
            ax.tricontour(tr, synthetic_values)
            success = True
        except ValueError:
            alpha += increment
        else:
            break
        if count > max_iterations:
            raise ValueError(
                f"Could not create a valid triangulation mask within {max_iterations}"
            )
    plt.close(fig)
    triang.set_mask(maxi > alpha)
    return triang


def lighten_color(color: str, amount: float = 0.5) -> Tuple[float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*to_rgb(c)))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_typology_day(
    utci: pd.Series,
    dbt: pd.Series,
    mrt: pd.Series,
    rh: pd.Series,
    ws: pd.Series,
    month: int = 6,
    day: int = 21,
    title: str = None,
) -> Figure:
    """Plot a single days UTCI with composite DBT, RH, MRT and WS components shown also

    Args:
        utci (pd.Series): An annual time-indexed series containing UTCI values.
        dbt (pd.Series): An annual time-indexed series containing DBT values.
        mrt (pd.Series): An annual time-indexed series containing MRT values.
        rh (pd.Series): An annual time-indexed series containing RH values.
        ws (pd.Series): An annual time-indexed series containing WS values.
        month (int, optional): The month to plot.
        day (int, optional): The day to plot.

    Returns:
        Figure: A matplotlib Figure object.
    """

    if any([all(utci.index != i.index) for i in [dbt, mrt, rh, ws]]):
        raise ValueError("All series must have the same index")

    try:
        dt = f"{utci.index.year[0]}-{month}-{day}"
        date = utci.loc[dt].index[0]
    except KeyError as e:
        raise e

    fig, ax = plt.subplots(figsize=(10, 4))

    axes = []
    for i in range(5):
        if i == 0:
            axes.append(ax)
        else:
            temp_ax = ax.twinx()
            rspine = temp_ax.spines["right"]
            rspine.set_position(("axes", 0.88 + (0.12 * i)))
            temp_ax.set_frame_on(True)
            temp_ax.patch.set_visible(False)
            axes.append(temp_ax)

    (a,) = axes[0].plot(utci.loc[dt], c="black", label="UTCI")
    axes[0].set_ylabel("UTCI")
    (b,) = axes[1].plot(dbt.loc[dt], c="red", alpha=0.75, label="DBT", ls="--")
    axes[1].set_ylabel("DBT")
    (c,) = axes[2].plot(mrt.loc[dt], c="orange", alpha=0.75, label="MRT", ls="--")
    axes[2].set_ylabel("MRT")
    (d,) = axes[3].plot(rh.loc[dt], c="blue", alpha=0.75, label="RH", ls="--")
    axes[3].set_ylabel("RH")
    (e,) = axes[4].plot(ws.loc[dt], c="green", alpha=0.75, label="WS", ls="--")
    axes[4].set_ylabel("WS")

    [[ax.spines[spine].set_visible(False) for spine in ["top"]] for ax in axes]
    [axes[0].spines[spine].set_visible(False) for spine in ["right"]]
    [axes[0].spines[j].set_color("k") for j in ["bottom", "left"]]

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set_xlim(utci.loc[dt].index.min(), utci.loc[dt].index.max())

    lgd = axes[0].legend(
        handles=[a, b, c, d, e],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=[0.5, -0.25],
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    [plt.setp(text, color="k") for text in lgd.get_texts()]

    if title:
        ax.set_title(f"{date:%B %d} - {title}", ha="left", va="bottom", x=0)

    plt.tight_layout()

    return fig


def plot_spatial(
    triangulations: List[Triangulation],
    values: List[List[float]],
    levels: Union[List[float], int] = None,
    colormap: Colormap = "viridis",
    extend: str = "neither",
    norm: BoundaryNorm = None,
    xlims: List[float] = None,
    ylims: List[float] = None,
    colorbar_label: str = "",
    title: str = "",
) -> Figure:
    """Plot a spatial map of a variable using a triangulation and associated values.

    Args:
        triangulations (List[Triangulation]): A list of triangulations to plot.
        values (List[List[float]]): A list of values, corresponding with the triangulations and their respective indices.
        levels (Union[List[float], int], optional): The number of levels to include in the colorbar. Defaults to None which will use 10-steps between the min/max for all given values.
        colormap (Colormap, optional): The . Defaults to "viridis".
        extend (str, optional): Define how to handle the end-points of the colorbar. Defaults to "neither".
        norm (BoundaryNorm, optional): A matploltib BoundaryNorm object containing colormap boundary mapping information. Defaults to None.
        xlims (List[float], optional): The x-limit for the plot. Defaults to None.
        ylims (List[float], optional): The y-limit for the plot. Defaults to None.
        colorbar_label (str, optional): A label to be placed next to the colorbar. Defaults to "".
        title (str, optional): The title to be placed on the plot. Defaults to "".

    Returns:
        Figure: A matplotlib Figure object.
    """
    for tri, zs in list(zip(*[triangulations, values])):
        if len(tri.x) != len(zs):
            raise ValueError(
                "The shape of the triangulations and values given do not match."
            )

    if levels is None:
        levels = np.linspace(
            min([np.amin(i) for i in values]), max([np.amax(i) for i in values]), 10
        )

    if xlims is None:
        xlims = [
            min([i.x.min() for i in triangulations]),
            max([i.x.max() for i in triangulations]),
        ]

    if ylims is None:
        ylims = [
            min([i.y.min() for i in triangulations]),
            max([i.y.max() for i in triangulations]),
        ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    for tri, zs in list(zip(*[triangulations, values])):
        tcf = ax.tricontourf(
            tri, zs, extend=extend, cmap=colormap, levels=levels, norm=norm
        )

    # Plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)

    cbar = plt.colorbar(tcf, cax=cax, format=mticker.StrMethodFormatter("{x:04.1f}"))
    cbar.outline.set_visible(False)
    cbar.set_label(colorbar_label)

    ax.set_title(title, ha="left", va="bottom", x=0)

    plt.tight_layout()

    return fig


def plot_heatmap(
    collection: HourlyContinuousCollection,
    colormap: Colormap = "viridis",
    norm: BoundaryNorm = None,
    vlims: List[float] = None,
    title: str = None,
) -> Figure:
    """Plot a heatmap for a given Ladybug HourlyContinuousCollection.

    Args:
        collection (HourlyContinuousCollection): A Ladybug HourlyContinuousCollection object.
        colormap (Colormap, optional): The colormap to use in this heatmap. Defaults to "viridis".
        norm (BoundaryNorm, optional): A matplotlib BoundaryNorm object describing value thresholds. Defaults to None.
        vlims (List[float], optional): The limits to which values should be plotted (useful for comparing between different cases). Defaults to None.
        title (str, optional): A title to place at the top of the plot. Defaults to None.

    Returns:
        Figure: A matplotlib Figure object.
    """
    series = to_series(collection)

    if norm and vlims:
        raise ValueError("You cannot pass both vlims and a norm value to this method.")

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Reshape data into time/day matrix
    day_time_matrix = (
        series.to_frame()
        .pivot_table(columns=series.index.date, index=series.index.time)
        .values[::-1]
    )

    # Plot data
    heatmap = ax.imshow(
        day_time_matrix,
        extent=[
            mdates.date2num(series.index.min()),
            mdates.date2num(series.index.max()),
            726449,
            726450,
        ],
        aspect="auto",
        cmap=colormap,
        norm=norm,
        interpolation="none",
        vmin=None if vlims is None else vlims[0],
        vmax=None if vlims is None else vlims[1],
    )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha="left", color="k")
    plt.setp(ax.get_yticklabels(), color="k")

    [
        ax.spines[spine].set_visible(False)
        for spine in ["top", "bottom", "left", "right"]
    ]

    ax.grid(b=True, which="major", color="white", linestyle=":", alpha=1)

    cb = fig.colorbar(
        heatmap,
        orientation="horizontal",
        drawedges=False,
        fraction=0.05,
        aspect=100,
        pad=0.075,
    )
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")
    cb.outline.set_visible(False)

    if title is None:
        ax.set_title(series.name, color="k", y=1, ha="left", va="bottom", x=0)
    else:
        ax.set_title(
            f"{series.name} - {title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    # Tidy plot
    plt.tight_layout()

    return fig


def plot_utci_heatmap_histogram(
    collection: HourlyContinuousCollection, title: str = None
) -> Figure:
    """Create a histogram showing the annual hourly UTCI values associated with this Typology.

    Returns:
        Figure: A matplotlib Figure object.
    """

    assert (
        type(collection.header.data_type) == UniversalThermalClimateIndex
    ), f"Collection data type is not UTCI and cannot be used in this plot."

    # Instantiate figure
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    spec = fig.add_gridspec(
        ncols=1, nrows=2, width_ratios=[1], height_ratios=[4, 2], hspace=0.0
    )
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    divider = make_axes_locatable(histogram_ax)
    colorbar_ax = divider.append_axes("bottom", size="20%", pad=0.75)

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
        norm=UTCI_BOUNDARYNORM,
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
    heatmap_ax.grid(b=True, which="major", color="k", linestyle=":", alpha=0.5)

    # Add colorbar legend and text descriptors for comfort bands
    cb = fig.colorbar(
        heatmap,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=UTCI_LEVELS,
        drawedges=False,
    )
    cb.outline.set_visible(False)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")

    # Add labels to the colorbar
    levels = [-100] + UTCI_LEVELS + [100]
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

    # Add stacked plot
    t = pd.cut(series, [-100] + UTCI_LEVELS + [100], labels=UTCI_LABELS)
    t = t.groupby([t.index.month, t]).count().unstack().T
    t = t / t.sum()
    months = [
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
    ]
    t.T.plot.bar(
        ax=histogram_ax,
        stacked=True,
        color=[rgb2hex(UTCI_COLORMAP.get_under())]
        + UTCI_COLORMAP.colors
        + [rgb2hex(UTCI_COLORMAP.get_over())],
        legend=False,
        width=1,
    )
    histogram_ax.set_xlabel(None)
    histogram_ax.set_xlim(-0.5, 11.5)
    histogram_ax.set_ylim(0, 1)
    histogram_ax.set_xticklabels(months, ha="center", rotation=0, color="k")
    plt.setp(histogram_ax.get_yticklabels(), color="k")
    for spine in ["top", "right"]:
        histogram_ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        histogram_ax.spines[spine].set_color("k")
    histogram_ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))

    # # Add header percentages for bar plot
    cold_stress = t.T.filter(regex="Cold").sum(axis=1)
    heat_stress = t.T.filter(regex="Heat").sum(axis=1)
    no_stress = 1 - cold_stress - heat_stress
    for n, (i, j, k) in enumerate(zip(*[cold_stress, no_stress, heat_stress])):
        histogram_ax.text(
            n,
            1.02,
            "{0:0.1f}%".format(i * 100),
            va="bottom",
            ha="center",
            color="#3C65AF",
            fontsize="small",
        )
        histogram_ax.text(
            n,
            1.02,
            "{0:0.1f}%\n".format(j * 100),
            va="bottom",
            ha="center",
            color="#2EB349",
            fontsize="small",
        )
        histogram_ax.text(
            n,
            1.02,
            "{0:0.1f}%\n\n".format(k * 100),
            va="bottom",
            ha="center",
            color="#C31F25",
            fontsize="small",
        )

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


def plot_utci_distance_to_comfortable(
    collection: HourlyContinuousCollection,
    title: str = None,
    comfort_thresholds: List[float] = [9, 26],
    low_limit: float = 15,
    high_limit: float = 25,
) -> Figure:
    """Plot the distance (in C) to comfortable for a given Ladybug HourlyContinuousCollection containing UTCI values.

    Args:
        collection (HourlyContinuousCollection): A Ladybug Universal Thermal Climate Index HourlyContinuousCollection object.
        title (str, optional): A title to place at the top of the plot. Defaults to None.
        comfort_thresholds (List[float], optional): The comfortable band of UTCI temperatures. Defaults to [9, 26].
        low_limit (float, optional): The distance from the lower edge of the comfort threshold to include in the "too cold" part of the heatmap. Defaults to 15.
        high_limit (float, optional): The distance from the upper edge of the comfort threshold to include in the "too hot" part of the heatmap. Defaults to 25.
    Returns:
        Figure: A matplotlib Figure object.
    """

    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError("This method only works for UTCI data.")

    if not len(comfort_thresholds) == 2:
        raise ValueError("comfort_thresholds must be a list of length 2.")

    # Create matrices containing the above/below/within UTCI distances to comfortable
    series = to_series(collection)

    low, high = comfort_thresholds
    midpoint = np.mean([low, high])

    distance_above_comfortable = (series[series > high] - high).to_frame()
    distance_above_comfortable_matrix = (
        distance_above_comfortable.set_index(
            [
                distance_above_comfortable.index.dayofyear,
                distance_above_comfortable.index.hour,
            ]
        )["Universal Thermal Climate Index (C)"]
        .astype(np.float64)
        .unstack()
        .T.reindex(range(24), axis=0)
        .reindex(range(365), axis=1)
    )

    distance_below_comfortable = (low - series[series < low]).to_frame()
    distance_below_comfortable_matrix = (
        distance_below_comfortable.set_index(
            [
                distance_below_comfortable.index.dayofyear,
                distance_below_comfortable.index.hour,
            ]
        )["Universal Thermal Climate Index (C)"]
        .astype(np.float64)
        .unstack()
        .T.reindex(range(24), axis=0)
        .reindex(range(365), axis=1)
    )

    distance_below_midpoint = (
        midpoint - series[(series >= low) & (series <= midpoint)]
    ).to_frame()
    distance_below_midpoint_matrix = (
        distance_below_midpoint.set_index(
            [
                distance_below_midpoint.index.dayofyear,
                distance_below_midpoint.index.hour,
            ]
        )["Universal Thermal Climate Index (C)"]
        .astype(np.float64)
        .unstack()
        .T.reindex(range(24), axis=0)
        .reindex(range(365), axis=1)
    )

    distance_above_midpoint = (
        series[(series <= high) & (series > midpoint)] - midpoint
    ).to_frame()
    distance_above_midpoint_matrix = (
        distance_above_midpoint.set_index(
            [
                distance_above_midpoint.index.dayofyear,
                distance_above_midpoint.index.hour,
            ]
        )["Universal Thermal Climate Index (C)"]
        .astype(np.float64)
        .unstack()
        .T.reindex(range(24), axis=0)
        .reindex(range(365), axis=1)
    )

    distance_above_comfortable_cmap = plt.get_cmap("YlOrRd")  # Reds
    distance_above_comfortable_lims = [0, high_limit]
    distance_above_comfortable_norm = BoundaryNorm(
        np.linspace(
            distance_above_comfortable_lims[0], distance_above_comfortable_lims[1], 100
        ),
        ncolors=distance_above_comfortable_cmap.N,
        clip=True,
    )

    distance_below_comfortable_cmap = plt.get_cmap("YlGnBu")  # Blues
    distance_below_comfortable_lims = [0, low_limit]
    distance_below_comfortable_norm = BoundaryNorm(
        np.linspace(
            distance_below_comfortable_lims[0], distance_below_comfortable_lims[1], 100
        ),
        ncolors=distance_below_comfortable_cmap.N,
        clip=True,
    )

    distance_below_midpoint_cmap = plt.get_cmap("YlGn_r")  # Greens_r
    distance_below_midpoint_lims = [0, midpoint - low]
    distance_below_midpoint_norm = BoundaryNorm(
        np.linspace(
            distance_below_midpoint_lims[0], distance_below_midpoint_lims[1], 100
        ),
        ncolors=distance_below_midpoint_cmap.N,
        clip=True,
    )

    distance_above_midpoint_cmap = plt.get_cmap("YlGn_r")  # Greens_r
    distance_above_midpoint_lims = [0, high - midpoint]
    distance_above_midpoint_norm = BoundaryNorm(
        np.linspace(
            distance_above_midpoint_lims[0], distance_above_midpoint_lims[1], 100
        ),
        ncolors=distance_above_midpoint_cmap.N,
        clip=True,
    )

    extent = [
        mdates.date2num(series.index.min()),
        mdates.date2num(series.index.max()),
        726449,
        726450,
    ]

    fig = plt.figure(constrained_layout=False, figsize=(15, 5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[20, 1])
    hmap_ax = fig.add_subplot(gs[0, :])
    cb_low_ax = fig.add_subplot(gs[1, 0])
    cb_mid_ax = fig.add_subplot(gs[1, 1])
    cb_high_ax = fig.add_subplot(gs[1, 2])

    distance_below_comfortable_plt = hmap_ax.imshow(
        np.ma.array(
            distance_below_comfortable_matrix,
            mask=np.isnan(distance_below_comfortable_matrix),
        ),
        extent=extent,
        aspect="auto",
        cmap=distance_below_comfortable_cmap,
        norm=distance_below_comfortable_norm,
        interpolation="none",
    )
    distance_below_midpoint_plt = hmap_ax.imshow(
        np.ma.array(
            distance_below_midpoint_matrix,
            mask=np.isnan(distance_below_midpoint_matrix),
        ),
        extent=extent,
        aspect="auto",
        cmap=distance_below_midpoint_cmap,
        norm=distance_below_midpoint_norm,
        interpolation="none",
    )
    distance_above_comfortable_plt = hmap_ax.imshow(
        np.ma.array(
            distance_above_comfortable_matrix,
            mask=np.isnan(distance_above_comfortable_matrix),
        ),
        extent=extent,
        aspect="auto",
        cmap=distance_above_comfortable_cmap,
        norm=distance_above_comfortable_norm,
        interpolation="none",
    )
    distance_above_midpoint_plt = hmap_ax.imshow(
        np.ma.array(
            distance_above_midpoint_matrix,
            mask=np.isnan(distance_above_midpoint_matrix),
        ),
        extent=extent,
        aspect="auto",
        cmap=distance_above_midpoint_cmap,
        norm=distance_above_midpoint_norm,
        interpolation="none",
    )

    # Axis formatting
    hmap_ax.invert_yaxis()
    hmap_ax.xaxis_date()
    hmap_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    hmap_ax.yaxis_date()
    hmap_ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    hmap_ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(hmap_ax.get_xticklabels(), ha="left", color="k")
    plt.setp(hmap_ax.get_yticklabels(), color="k")

    # Spine formatting
    [
        hmap_ax.spines[spine].set_visible(False)
        for spine in ["top", "bottom", "left", "right"]
    ]

    # Grid formatting
    hmap_ax.grid(visible=True, which="major", color="white", linestyle=":", alpha=1)

    # Colorbars
    low_cb = ColorbarBase(
        cb_low_ax,
        cmap=distance_below_comfortable_cmap,
        orientation="horizontal",
        norm=distance_below_comfortable_norm,
        label='Degrees below "comfortable"',
        extend="max",
    )
    low_cb.outline.set_visible(False)
    cb_low_ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    mid_cb = ColorbarBase(
        cb_mid_ax,
        cmap=distance_below_midpoint_cmap,
        orientation="horizontal",
        norm=distance_below_midpoint_norm,
        label='Degrees about "comfortable"',
        extend="neither",
    )
    mid_cb.outline.set_visible(False)
    cb_mid_ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    high_cb = ColorbarBase(
        cb_high_ax,
        cmap=distance_above_comfortable_cmap,
        orientation="horizontal",
        norm=distance_above_comfortable_norm,
        label='Degrees above "comfortable"',
        extend="max",
    )
    high_cb.outline.set_visible(False)
    cb_high_ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    if title is None:
        hmap_ax.set_title(
            'Distance to "comfortable"', color="k", y=1, ha="left", va="bottom", x=0
        )
    else:
        hmap_ax.set_title(
            "{0:} - {1:}".format('Distance to "comfortable"', title),
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    # Tidy plot
    plt.tight_layout()

    return fig


def plot_utci_journey(
    utci_values: List[float],
    title: str,
    names: List[str] = None,
    curve: bool = False,
    show_legend: bool = True,
    rotation: float = 0,
    yscale: float = -1
) -> Figure:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a given time of year

    Args:
        utci_values (float): A list of UTCI values.
        names (List[str], optional): A list of names to label each value with. Defaults to None.
        curve (bool, optional): Whether to plot the pseudo-journey as a spline. Defaults to False.
        show_legend (bool, optional): Set to True to plot the UTCI comfort band legend also.
        rotation (float, optional): Rotates the label of each value by a specified number of degrees. 

    Returns:
        Figure: A matplotlib figure object.
    """

    if names:
        if len(utci_values) != len(names):
            raise ValueError("Number of values and names must be equal.")
    else:
        names = [str(i) for i in range(len(utci_values))]

    # Convert collections into series and combine
    df_pit = pd.Series(utci_values, index=names)

    fig, ax = plt.subplots(figsize=(10 , 2.5))
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        ax.text(
            n, val, idx, c="k", zorder=10, ha="center", va="center", fontsize="medium", rotation=rotation
        )

    middf = 0.5 * (utci_values.max() + utci_values.min())

    if yscale == -1:
        yscale = 1.5 * (utci_values.max() - middf)

    ax.set_ylim(middf - yscale, middf + yscale)

    categories = [
        " ".join([i.title() for i in p.split("_")]) for p in [
            "extreme_cold_stress", 
            "very_strong_cold_stress", 
            "strong_cold_stress", 
            "moderate_cold_stress", 
            "slight_cold_stress", 
            "no_thermal_stress", 
            "moderate_heat_stress", 
            "strong_heat_stress", 
            "very_strong_heat_stress", 
            "extreme_heat_stress"
        ]
    ]
    utci_handles = []
    low_edit = UTCI_LEVELS[0:-1]
    high_edit = UTCI_LEVELS[1:]
    for low, high, color, category in list(zip(*[
        low_edit,
        high_edit,
        UTCI_COLORMAP.colors,
        categories
        ]
    )
    ):
        utci_handles.append(mpatches.Patch(color=color, label=category))
        ax.axhspan(low, high, color=color, alpha=0.4, lw=0)

    utci_handles.append(mpatches.Patch(color="#0D104B", label=categories[0]))
    ax.axhspan(-99, UTCI_LEVELS[0], color="#0D104B", alpha=0.4, lw=0)

    utci_handles.append(mpatches.Patch(color="#580002", label=categories[-1]))
    ax.axhspan(UTCI_LEVELS[-1], 99, color="#580002", alpha=0.4, lw=0)

    if curve:
        # Smooth values
        if len(utci_values) < 3:
            k = 1
        else:
            k = 2
        x = np.arange(len(utci_values))
        y = df_pit.values
        xnew = np.linspace(min(x), max(x), 300)
        bspl = make_interp_spline(x, y, k=k)
        ynew = bspl(xnew)

        # Plot the smoothed values
        ax.plot(xnew, ynew, c="#B30202", ls="--", zorder=3)

    [ax.spines[spine].set_visible(False) for spine in ["top", "right", "bottom"]]
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    ax.set_ylabel("UTCI (°C)")
    ax.set_title(
        title,
        color="k",
        y=1,
        ha="left",
        va="bottom",
        x=0,
    )
    plt.tight_layout()

    return fig


def utci_comparison_diurnal(
    collections: List[HourlyContinuousCollection],
    collection_ids: List[str] = None,
) -> Figure:
    """Plot a set of UTCI collections on a single figure for monthly diurnal periods.

    Args:
        collections (List[HourlyContinuousCollection]): A list of UTCI collections.
        collection_ids (List[str], optional): A list of descriptions for each of the input collections. Defaults to None.

    Returns:
        Figure: A matplotlib figure object.
    """

    if collection_ids is None:
        collection_ids = [f"{i:02d}" for i in range(len(collections))]
    assert len(collections) == len(
        collection_ids
    ), "The length of collections_ids must match the number of collections."

    for n, col in enumerate(collections):
        assert (
            type(col.header.data_type) == UniversalThermalClimateIndex
        ), f"Collection {n} data type is not UTCI and cannot be used in this plot."

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

    x_values = range(288)
    idx = [
        item
        for sublist in [
            [datetime(2021, month, 1, hour, 0, 0) for hour in range(24)]
            for month in range(1, 13)
        ]
        for item in sublist
    ]
    idx_str = [i.strftime("%b %H:%M") for i in idx]

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
        [ax.spines[spine].set_visible(False) for spine in ["top", "right"]]
        [ax.spines[j].set_color("k") for j in ["bottom", "left"]]

    # Get plotted values attributes
    ylim = ax.get_ylim()
    mitigation_handles, mitigation_labels = ax.get_legend_handles_labels()

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
            utci_handles.append(patches.Patch(color=cc, label=category))
        ax.grid(b=True, which="major", axis="both", c="k", ls="--", lw=1, alpha=0.1)
        ax.grid(b=True, which="minor", axis="x", c="k", ls=":", lw=1, alpha=0.1)
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
    lgd = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=[1, 0.9],
        frameon=False,
        fontsize="small",
        ncol=1,
    )

    ti = fig.suptitle(
        f"Average diurnal profile",
        ha="left",
        va="bottom",
        x=0.05,
        y=0.95,
    )

    plt.tight_layout()

    return fig
