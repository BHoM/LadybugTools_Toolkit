from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...bhomutil.analytics import CONSOLE_LOGGER
from ...ladybug_extension.analysis_period import describe_analysis_period
from ...plot.utilities import create_triangulation
from .spatial_comfort import SpatialComfort, SpatialMetric


def get_common_pt_indices(
    spatial_result_1: SpatialComfort, spatial_result_2: SpatialComfort
) -> List[List[int]]:
    """Get the indices of the points that are common between two spatial comfort cases.

    Args:
        spatial_result_1 (SpatialComfort):
            A SpatialComfort object.
        spatial_result_2 (SpatialComfort):
            A SpatialComfort object.

    Returns (List[List[int]]):
        A list of two lists of indices that correspond to the points in the two.

    """
    return pd.merge(
        spatial_result_1.points.reset_index()[["index", "x", "y"]],
        spatial_result_2.points.reset_index()[["index", "x", "y"]],
        on=["x", "y"],
    )[["index_x", "index_y"]].values.T


def get_triangulation(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    alpha: float = 1.9,
) -> Triangulation:
    """Get a triangulation of the points that are common between two spatial comfort cases.

    Args:
        spatial_result_1 (SpatialComfort):
            A SpatialComfort object.
        spatial_result_2 (SpatialComfort):
            A SpatialComfort object.
        alpha (float, optional):
            A number between 0 and 2 that controls the distance between vertices of the triangulation, above which cells are removed. Default is 1.9.

    Returns (Triangulation):
        A matplotlib Triangulation object.
    """

    sc1_idx, _ = get_common_pt_indices(spatial_result_1, spatial_result_2)
    xx = spatial_result_1.points.iloc[sc1_idx].x.values
    yy = spatial_result_1.points.iloc[sc1_idx].y.values
    return create_triangulation(xx, yy, alpha=alpha)


def compare_utci(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    metric: SpatialMetric,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    alpha: float = 1.9,
) -> plt.Figure:
    """Plot the difference in typical UTCI between two spatial comfort cases.

    Args:
        spatial_result_1 (SpatialComfort):
            A SpatialComfort object.
        spatial_result_2 (SpatialComfort):
            A SpatialComfort object.
        metric (SpatialMetric):
            The metric to be plotted. Can be either SpatialMetric.UTCI_INTERPOLATED or SpatialMetric.UTCI_CALCULATED.
        analysis_period (AnalysisPeriod, optional):
            The analysis period overwhich comparison shoudl be made. Defaults to AnalysisPeriod().
        alpha (float, optional):
            An alpha value from which to create the triangulation between point locations. Defaults to 1.9.

    Raises:
        ValueError: _description_

    Returns:
        plt.Figure: _description_
    """

    if metric.value not in [
        SpatialMetric.UTCI_INTERPOLATED.value,
        SpatialMetric.UTCI_CALCULATED.value,
    ]:
        raise ValueError("This type of plot is not possible for the requested metric.")

    CONSOLE_LOGGER.info(
        f"[SpatialComfort - Comparison] - Plotting UTCI difference between {spatial_result_1} and {spatial_result_2} for {describe_analysis_period(analysis_period)}"
    )
    # get point indices
    indices = get_common_pt_indices(spatial_result_1, spatial_result_2)

    # create masks
    index_mask = [i in analysis_period.hoys_int for i in range(8760)]
    col_mask_1 = [i in indices[0] for i in spatial_result_1.points.index]
    col_mask_2 = [i in indices[1] for i in spatial_result_2.points.index]

    # calculate typical UTCI
    utci_1 = (
        spatial_result_1.get_spatial_metric(metric)
        .iloc[index_mask, col_mask_1]
        .mean(axis=0)
        .reset_index(drop=True)
    )
    utci_2 = (
        spatial_result_2.get_spatial_metric(metric)
        .iloc[index_mask, col_mask_2]
        .mean(axis=0)
        .reset_index(drop=True)
    )

    # obtain difference
    utci_diff = utci_2 - utci_1

    # triangulate
    tri = get_triangulation(spatial_result_1, spatial_result_2, alpha=alpha)

    # plot heatmap
    tcf_properties = {
        "cmap": "RdBu_r",
        "levels": np.linspace(-10, 10, 11),
        "extend": "both",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim([min(spatial_result_1.points.x), max(spatial_result_1.points.x)])
    ax.set_ylim([min(spatial_result_1.points.y), max(spatial_result_1.points.y)])

    # add contour-fill
    tcf = ax.tricontourf(tri, utci_diff.values, **tcf_properties)

    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
    cbar = plt.colorbar(tcf, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Typical change in UTCI (C)")

    # add title
    ax.set_title(
        f"{describe_analysis_period(analysis_period)}\n{metric.description()} - average difference\n{spatial_result_1} > {spatial_result_2}",
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

    return fig


def compare_mrt(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    alpha: float = 1.9,
) -> plt.Figure:
    """Compare two spatial comfort cases. WARNING: This can use A LOT of memory as it must load two sets of results to compare them.

    Args:
        spatial_result_1 (SpatialComfort):
            The first SpatialComfort object.
        spatial_result_2 (SpatialComfort):
            The second SpatialComfort object.
        analysis_period (AnalysisPeriod, optional):
            The analysis period to use for the comparison. Defaults to AnalysisPeriod().
        alpha (float, optional):
            An alpha value to use for the triangulation. Defaults to 1.9.

    Returns:
        plt.Figure:
            A matplotlib figure.
    """

    CONSOLE_LOGGER.info(
        f"[SpatialComfort - Comparison] - Plotting MRT difference between {spatial_result_1} and {spatial_result_2} for {describe_analysis_period(analysis_period)}"
    )
    # get point indices
    indices = get_common_pt_indices(spatial_result_1, spatial_result_2)

    # create masks
    index_mask = [i in analysis_period.hoys_int for i in range(8760)]
    col_mask_1 = [i in indices[0] for i in spatial_result_1.points.index]
    col_mask_2 = [i in indices[1] for i in spatial_result_2.points.index]

    metric = SpatialMetric.MRT_INTERPOLATED

    # calculate typical MRT
    mrt_1 = (
        spatial_result_1.get_spatial_metric(metric)
        .iloc[index_mask, col_mask_1]
        .mean(axis=0)
        .reset_index(drop=True)
    )
    mrt_2 = (
        spatial_result_2.get_spatial_metric(metric)
        .iloc[index_mask, col_mask_2]
        .mean(axis=0)
        .reset_index(drop=True)
    )

    # obtain difference
    mrt_diff = mrt_2 - mrt_1

    # triangulate
    tri = get_triangulation(spatial_result_1, spatial_result_2, alpha=alpha)

    # plot heatmap
    tcf_properties = {
        "cmap": "PuOr_r",
        "levels": np.linspace(-10, 10, 11),
        "extend": "both",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(
        [min(spatial_result_1.points.x.values), max(spatial_result_1.points.x.values)]
    )
    ax.set_ylim(
        [min(spatial_result_1.points.y.values), max(spatial_result_1.points.y.values)]
    )

    # add contour-fill
    tcf = ax.tricontourf(tri, mrt_diff.values, **tcf_properties)

    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
    cbar = plt.colorbar(tcf, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Typical change in MRT (C)")

    # add title
    ax.set_title(
        f"{describe_analysis_period(analysis_period)}\n{metric.description()} - average difference\n{spatial_result_1} > {spatial_result_2}",
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

    return fig


def compare_sun_hours(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    alpha: float = 1.9,
) -> plt.Figure:
    """Compare two spatial comfort cases. WARNING: This can use A LOT of memory as it must load two sets of results to compare them.

    Args:
        spatial_result_1 (SpatialComfort):
            The first SpatialComfort object.
        spatial_result_2 (SpatialComfort):
            The second SpatialComfort object.
        analysis_period (AnalysisPeriod, optional):
            The analysis period to use for the comparison. Defaults to AnalysisPeriod().
        alpha (float, optional):
            An alpha value to use for the triangulation. Defaults to 1.9.

    Returns:
        plt.Figure:
            A matplotlib figure.
    """

    CONSOLE_LOGGER.info(
        f"[SpatialComfort - Comparison] - Plotting Sun-Hours difference between {spatial_result_1} and {spatial_result_2} for {describe_analysis_period(analysis_period)}"
    )
    # get point indices
    indices = get_common_pt_indices(spatial_result_1, spatial_result_2)

    # create masks
    col_mask_1 = [i in indices[0] for i in spatial_result_1.points.index]
    col_mask_2 = [i in indices[1] for i in spatial_result_2.points.index]

    metric = SpatialMetric.DIRECT_SUN_HOURS

    # calculate typical sun-hours
    sh_1 = spatial_result_1.direct_sun_hours(analysis_period).iloc[col_mask_1].values
    sh_2 = spatial_result_2.direct_sun_hours(analysis_period).iloc[col_mask_2].values

    # obtain difference
    sh_diff = sh_2 - sh_1

    # triangulate
    tri = get_triangulation(spatial_result_1, spatial_result_2, alpha=alpha)

    # plot heatmap
    tcf_properties = {
        "cmap": "RdGy_r",
        "levels": np.linspace(-2, 2, 11),
        "extend": "both",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(
        [min(spatial_result_1.points.x.values), max(spatial_result_1.points.x.values)]
    )
    ax.set_ylim(
        [min(spatial_result_1.points.y.values), max(spatial_result_1.points.y.values)]
    )

    # add contour-fill
    tcf = ax.tricontourf(tri, sh_diff, **tcf_properties)

    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
    cbar = plt.colorbar(tcf, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Typical change in Direct Sun Hours (hours)")

    # add title
    ax.set_title(
        f"{describe_analysis_period(analysis_period)}\n{metric.description()} - average difference\n{spatial_result_1} > {spatial_result_2}",
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

    return fig


def compare_ws(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    metric: SpatialMetric,
    analysis_period: AnalysisPeriod = None,
    alpha: float = 1.9,
) -> plt.Figure:
    """Compare two spatial comfort cases. WARNING: This can use A LOT of memory as it must load two sets of results to compare them.

    Args:
        spatial_result_1 (SpatialComfort):
            The first SpatialComfort object.
        spatial_result_2 (SpatialComfort):
            The second SpatialComfort object.
        metric (SpatialMetric):
            The metric to compare.
        analysis_period (AnalysisPeriod, optional):
            The analysis period to use for the comparison. Defaults to AnalysisPeriod().
        alpha (float, optional):
            An alpha value to use for the triangulation. Defaults to 1.9.

    Returns:
        plt.Figure:
            A matplotlib figure.
    """

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    if metric.value not in [
        SpatialMetric.WS_CFD.value,
    ]:
        raise ValueError("This type of plot is not possible for the requested metric.")

    CONSOLE_LOGGER.info(
        f"[SpatialComfort - Comparison] - Plotting WS difference between {spatial_result_1} and {spatial_result_2} for {describe_analysis_period(analysis_period)}"
    )
    # get point indices
    indices = get_common_pt_indices(spatial_result_1, spatial_result_2)

    # create masks
    index_mask = [i in analysis_period.hoys_int for i in range(8760)]
    col_mask_1 = [i in indices[0] for i in spatial_result_1.points.index]
    col_mask_2 = [i in indices[1] for i in spatial_result_2.points.index]

    # calculate typical WS
    ws_1 = (
        spatial_result_1.get_spatial_metric(metric)
        .iloc[index_mask, col_mask_1]
        .mean(axis=0)
        .reset_index(drop=True)
    )
    ws_2 = (
        spatial_result_2.get_spatial_metric(metric)
        .iloc[index_mask, col_mask_2]
        .mean(axis=0)
        .reset_index(drop=True)
    )

    # obtain difference
    ws_diff = ws_2 - ws_1

    # triangulate
    tri = get_triangulation(spatial_result_1, spatial_result_2, alpha=alpha)

    # plot heatmap
    tcf_properties = {
        "cmap": "PRGn_r",
        "levels": np.linspace(-4, 4, 9),
        "extend": "both",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(
        [min(spatial_result_1.points.x.values), max(spatial_result_1.points.x.values)]
    )
    ax.set_ylim(
        [min(spatial_result_1.points.y.values), max(spatial_result_1.points.y.values)]
    )

    # add contour-fill
    tcf = ax.tricontourf(tri, ws_diff.values, **tcf_properties)

    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
    cbar = plt.colorbar(tcf, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Typical change in Wind Speed (m/s)")

    # add title
    ax.set_title(
        f"{describe_analysis_period(analysis_period)}\n{metric.description()} - average difference\n{spatial_result_1} > {spatial_result_2}",
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

    return fig


def compare_distance_to_comfortable(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    metric: SpatialMetric,
    analysis_period: AnalysisPeriod = None,
    comfort_limits: Tuple[float] = (9, 26),
    alpha: float = 1.9,
) -> plt.Figure:
    """For a given metric, plot the difference between two SpatialComfort objects.

    Args:
        spatial_result_1 (SpatialComfort):
            The first SpatialComfort object to compare.
        spatial_result_2 (SpatialComfort):
            The second SpatialComfort object to compare.
        metric (SpatialMetric):
            The metric to compare.
        analysis_period (AnalysisPeriod, optional):
            An optional analysis period to limit the comparison to. Defaults to None.
        comfort_limits (Tuple[float], optional):
            The comfort limits to use for the comparison. Defaults to (9, 26).
        alpha (float, optional):
            An alpha value to use for triangulation. Defaults to 1.9.

    Returns:
        plt.Figure:
            A matplotlib figure object.
    """

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    if metric.value not in [
        SpatialMetric.UTCI_INTERPOLATED.value,
        SpatialMetric.UTCI_CALCULATED.value,
    ]:
        raise ValueError("This type of plot is not possible for the requested metric.")

    # get comfort midpoint
    comfort_mid = np.mean(comfort_limits)

    # get point indices
    indices = get_common_pt_indices(spatial_result_1, spatial_result_2)

    # create masks
    index_mask = [i in analysis_period.hoys_int for i in range(8760)]
    col_mask_1 = [i in indices[0] for i in spatial_result_1.points.index]
    col_mask_2 = [i in indices[1] for i in spatial_result_2.points.index]

    # filter for target time period and calculate distance to comfort midband
    comfort_distance_1 = (
        abs(
            spatial_result_1.get_spatial_metric(metric).iloc[index_mask, col_mask_1]
            - comfort_mid
        )
        .T.reset_index(drop=True)
        .T
    )
    comfort_distance_2 = (
        abs(
            spatial_result_2.get_spatial_metric(metric).iloc[index_mask, col_mask_2]
            - comfort_mid
        )
        .T.reset_index(drop=True)
        .T
    )

    # get difference between comfort distances
    comfort_diff = comfort_distance_2 - comfort_distance_1

    # triangulate
    tri = get_triangulation(spatial_result_1, spatial_result_2, alpha=alpha)

    # plot heatmap
    tcf_properties = {
        "cmap": "BrBG",
        "levels": np.linspace(-5, 5, 11),
        "extend": "both",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(
        [min(spatial_result_1.points.x.values), max(spatial_result_1.points.x.values)]
    )
    ax.set_ylim(
        [min(spatial_result_1.points.y.values), max(spatial_result_1.points.y.values)]
    )

    # add contour-fill
    tcf = ax.tricontourf(tri, -comfort_diff.mean(axis=0).values, **tcf_properties)

    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, aspect=20)
    cbar = plt.colorbar(tcf, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Change in degrees from comfort midband (C)")

    # add title
    ax.set_title(
        f"{describe_analysis_period(analysis_period)}\n{metric.description()} - average difference\n{spatial_result_1} > {spatial_result_2}",
        ha="left",
        va="bottom",
        x=0,
    )

    plt.tight_layout()

    return fig
