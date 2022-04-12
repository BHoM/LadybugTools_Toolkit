from __future__ import annotations

import copy
from typing import List, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug_extension.datacollection import to_series
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    is_color_like,
    rgb2hex,
)
from matplotlib.figure import Figure
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline


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


def create_triangulation(x: List[float], y: List[float]) -> Triangulation:
    """Create a matplotlib Triangulation from a list of x and y coordinates, including a mask to remove islands within this triangulation.

    Args:
        x (List[float]): A list of x coordinates.
        y (List[float]): A list of y coordinates.

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
    max_iterations = 100
    alpha = 0.5
    increment = 0.02
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
        ax.set_title(f"{title} - {date:%B %d}", ha="left", va="bottom", x=0)

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


def plot_utci_distance_to_comfortable(
    collection: HourlyContinuousCollection,
    title: str = None,
    comfort_thresholds: List[float] = [9, 26],
    distance_from_comfort_to_show: float = 20,
) -> Figure:
    """Plot the distance (in C) to comfortable for a given Ladybug HourlyContinuousCollection containing UTCI values.

    Args:
        collection (HourlyContinuousCollection): A Ladybug Universal Thermal Climate Index HourlyContinuousCollection object.
        title (str, optional): A title to place at the top of the plot. Defaults to None.
        comfort_thresholds (List[float], optional): The comfortable band of UTCI temperatures. Defaults to [9, 26].
        distance_from_comfort_to_show (float, optional): The range of temperatures to include in this plot. Defaults to 20.

    Returns:
        Figure: A matplotlib Figure object.
    """
    raise NotImplementedError("This method not yet ready for distribution.")

    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError("This method only works for UTCI data.")

    if not len(comfort_thresholds) == 2:
        raise ValueError("The comfort_thresholds must be a list of length 2.")

    # Create colormap and boundarynorm
    v_mid = sum(comfort_thresholds) / 2
    v_low = v_mid - distance_from_comfort_to_show
    v_high = v_mid + distance_from_comfort_to_show

    colormap = LinearSegmentedColormap.from_list(
        "_",
        list(zip([0, 0.25, 0.5, 0.75, 1], ["blue", "white", "green", "white", "red"])),
        N=256,
    )
    norm = BoundaryNorm(np.linspace(v_low, v_high, 101), ncolors=colormap.N, clip=True)

    series = to_series(collection)
    matrix = (
        series.to_frame()
        .pivot_table(columns=series.index.date, index=series.index.time)
        .values[::-1]
    )
    distance_to_comfort_midpoint = matrix - v_mid

    # Plot infrastructure
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    heatmap = ax.imshow(
        distance_to_comfort_midpoint,
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
    )

    # Axis formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.tick_params(labelleft=True, labelright=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha="left", color="k")
    plt.setp(ax.get_yticklabels(), color="k")

    # Spine formatting
    [
        ax.spines[spine].set_visible(False)
        for spine in ["top", "bottom", "left", "right"]
    ]

    # Grid formatting
    ax.grid(visible=True, which="major", color="white", linestyle=":", alpha=1)

    # Colorbar formatting
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="6%", pad=0.4, pack_start=True)
    cb = ColorbarBase(
        ax_cb,
        cmap=colormap,
        orientation="horizontal",
        norm=norm,
    )
    plt.gcf().add_axes(ax_cb)

    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="k")
    cb.outline.set_visible(False)

    # ax_cb.text(
    #     v_mid,
    #     0,
    #     "Comfortable",
    #     color="k",
    #     fontsize="small",
    #     ha="center",
    #     va="top",
    # )

    if title is None:
        ax.set_title(
            'Distance to "comfortable"', color="k", y=1, ha="left", va="bottom", x=0
        )
    else:
        ax.set_title(
            "{0:} - {1:}".format(series.name, title),
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    # Tidy plot
    plt.tight_layout()

    return fig


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


def plot_utci_journey(
    utci_values: List[float],
    names: List[str] = None,
    curve: bool = False,
    show_legend: bool = True,
) -> Figure:
    """Create a figure showing the pseudo-journey between different UTCI conditions at a given time of year

    Args:
        utci_values (float): A list of UTCI values.
        names (List[str], optional): A list of names to label each value with. Defaults to None.
        curve (bool, optional): Whether to plot the pseudo-journey as a spline. Defaults to False.
        show_legend (bool, optional): Set to True to plot the UTCI comfort band legend also.

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

    fig, ax = plt.subplots(figsize=(10, 2.5))
    for n, (idx, val) in enumerate(df_pit.items()):
        ax.scatter(n, val, c="white", s=400, zorder=9)
        ax.text(
            n, val, idx, c="k", zorder=10, ha="center", va="center", fontsize="medium"
        )

    ylims = ax.get_ylim()

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
    plt.tight_layout()

    return fig
