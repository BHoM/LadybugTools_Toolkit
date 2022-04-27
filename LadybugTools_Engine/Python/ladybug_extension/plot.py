from __future__ import annotations

from typing import List, Union

from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW, AnalysisPeriod
import numpy as np
from ladybug_extension.datacollection import to_series, to_array
from ladybug_extension.analysis_period import describe_analysis_period
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    Normalize,
)
from ladybug.windrose import Compass, WindRose
from matplotlib.figure import Figure

def heatmap(
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

    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

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

def rose(
    epw: EPW,
    collection: HourlyContinuousCollection,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    colormap: Union[Colormap, str] = "jet",
    norm: BoundaryNorm = None,
    directions: int = 12,
    bins: List[float] = None,
    hide_label_legend: bool = False,
    title: str = None,
) -> Figure:
    """Generate a windrose plot for the given wind directions and variable.

    Args:
        epw (EPW): A ladybug EPW object.
        collection (HourlyContinuousCollection): Annual hourly variables to bin and color.
        analysis_period (AnalysisPeriod, optional): An analysis period within which to assess the input data. Defaults to Annual.
        colormap (Colormap, optional): A colormap to apply to the binned data. Defaults to None.
        directions (int, optional): The number of directions to bin wind-direction into. Defaults to 12.
        value_bins (List[float], optional): A set of bins into which data will be binned. Defaults to None.
        hide_label_legend (bool, optional): Hide the label and legend. Defaults to False.

    Returns:
        Figure: A matplotlib figure object.
    """

    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    if bins is None:
        bins = np.linspace(collection.min, collection.max, 11)

    colors = [colormap(i) for i in np.linspace(0, 1, len(bins))]

    ws_values = to_array(epw.wind_speed.filter_by_analysis_period(analysis_period))
    n_calm_hours = sum(ws_values == 0)

    not_calm: bool = to_array(epw.wind_speed) > 0
    filtered_collection = collection.filter_by_pattern(not_calm)
    filtered_wind_direction = epw.wind_direction.filter_by_pattern(not_calm)

    wr = WindRose(filtered_wind_direction, filtered_collection, directions)
    width = 360 / directions
    theta = np.radians(np.array(wr.angles) + (width / 2))[:-1]
    width = np.radians(width)

    binned_data = np.array([np.histogram(i, bins)[0] for i in wr.histogram_data])

    if title is None:
        title = "\n".join(
            [
                f"{to_series(collection).name} for {Path(epw.file_path).stem}",
                describe_analysis_period(analysis_period),
                f"Calm for {n_calm_hours / len(ws_values):0.2%} of the time ({n_calm_hours} hours)",
            ]
        )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.setp(ax.get_xticklabels(), fontsize="small")
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="both", ls="--")
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_xticks(np.radians(Compass.MAJOR_AZIMUTHS), minor=False)
    ax.set_xticklabels(Compass.MAJOR_TEXT, minor=False, **{"fontsize": "medium"})
    ax.set_xticks(np.radians(Compass.MINOR_AZIMUTHS), minor=True)
    ax.set_xticklabels(Compass.MINOR_TEXT, minor=True, **{"fontsize": "x-small"})

    bottom = np.zeros(directions)
    for n, d in enumerate(binned_data.T):
        ax.bar(x=theta, height=d, width=width, bottom=bottom, color=colors[n], ec=(1, 1, 1, 0.2), lw=0.5,)
        bottom += d

    if not hide_label_legend:
        norm = Normalize(vmin=bins[0], vmax=bins[-2]) if norm is None else norm
        colorbar_axes = fig.add_axes([1, 0.11, 0.03, 0.78]) 
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        colorbar = plt.colorbar(sm, ticks=bins, boundaries=bins, cax=colorbar_axes, label=to_series(collection).name)
        colorbar.outline.set_visible(False)

        ax.set_title(title, ha="left", x=0)

    plt.tight_layout()

    return fig