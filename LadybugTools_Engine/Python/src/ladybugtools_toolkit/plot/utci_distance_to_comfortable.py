from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


from ladybugtools_toolkit import analytics


@analytics
def utci_distance_to_comfortable(
    collection: HourlyContinuousCollection,
    title: str = None,
    comfort_thresholds: Tuple[float] = (9, 26),
    low_limit: float = 15,
    high_limit: float = 25,
) -> Figure:
    """Plot the distance (in C) to comfortable for a given Ladybug HourlyContinuousCollection
        containing UTCI values.

    Args:
        collection (HourlyContinuousCollection):
            A Ladybug Universal Thermal Climate Index HourlyContinuousCollection object.
        title (str, optional):
            A title to place at the top of the plot. Defaults to None.
        comfort_thresholds (List[float], optional):
            The comfortable band of UTCI temperatures. Defaults to [9, 26].
        low_limit (float, optional):
            The distance from the lower edge of the comfort threshold to include in the "too cold"
            part of the heatmap. Defaults to 15.
        high_limit (float, optional):
            The distance from the upper edge of the comfort threshold to include in the "too hot"
            part of the heatmap. Defaults to 25.
    Returns:
        Figure:
            A matplotlib Figure object.
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

    hmap_ax.imshow(
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
    hmap_ax.imshow(
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
    hmap_ax.imshow(
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
    hmap_ax.imshow(
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
    for spine in ["top", "bottom", "left", "right"]:
        hmap_ax.spines[spine].set_visible(False)

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
            f"Distance to comfortable - {title}",
            color="k",
            y=1,
            ha="left",
            va="bottom",
            x=0,
        )

    # Tidy plot
    plt.tight_layout()

    return fig
