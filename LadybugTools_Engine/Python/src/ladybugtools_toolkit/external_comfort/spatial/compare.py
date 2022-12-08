from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from matplotlib.tri.triangulation import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...bhomutil.analytics import CONSOLE_LOGGER
from ...ladybug_extension.analysis_period import describe as describe_analysis_period
from ...plot.create_triangulation import create_triangulation
from .spatial_comfort import SpatialComfort, SpatialMetric


def get_common_pt_indices(
    spatial_result_1: SpatialComfort, spatial_result_2: SpatialComfort
) -> List[List[int]]:
    """_"""
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
    """_"""
    sc1_idx, _ = get_common_pt_indices(spatial_result_1, spatial_result_2)
    xx = spatial_result_1.points.iloc[sc1_idx].x.values
    yy = spatial_result_1.points.iloc[sc1_idx].y.values
    return create_triangulation(xx, yy, alpha=alpha)


def compare_utci(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    metric: SpatialMetric,
    analysis_period: AnalysisPeriod = None,
    alpha: float = 1.9,
) -> plt.Figure:
    """_"""

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

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
        spatial_result_1._get_spatial_metric(metric)
        .iloc[index_mask, col_mask_1]
        .mean(axis=0)
        .reset_index(drop=True)
    )
    utci_2 = (
        spatial_result_2._get_spatial_metric(metric)
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
    ax.set_xlim([min(spatial_result_1._points_x), max(spatial_result_1._points_x)])
    ax.set_ylim([min(spatial_result_1._points_y), max(spatial_result_1._points_y)])

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
    analysis_period: AnalysisPeriod = None,
    alpha: float = 1.9,
) -> plt.Figure:
    """_"""

    if analysis_period is None:
        analysis_period = AnalysisPeriod()

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
        spatial_result_1._get_spatial_metric(metric)
        .iloc[index_mask, col_mask_1]
        .mean(axis=0)
        .reset_index(drop=True)
    )
    mrt_2 = (
        spatial_result_2._get_spatial_metric(metric)
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
        "cmap": "PiYG_r",
        "levels": np.linspace(-10, 10, 11),
        "extend": "both",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim([min(spatial_result_1._points_x), max(spatial_result_1._points_x)])
    ax.set_ylim([min(spatial_result_1._points_y), max(spatial_result_1._points_y)])

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


def compare_ws(
    spatial_result_1: SpatialComfort,
    spatial_result_2: SpatialComfort,
    metric: SpatialMetric,
    analysis_period: AnalysisPeriod = None,
    alpha: float = 1.9,
) -> plt.Figure:
    """_"""

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
        spatial_result_1._get_spatial_metric(metric)
        .iloc[index_mask, col_mask_1]
        .mean(axis=0)
        .reset_index(drop=True)
    )
    ws_2 = (
        spatial_result_2._get_spatial_metric(metric)
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
    ax.set_xlim([min(spatial_result_1._points_x), max(spatial_result_1._points_x)])
    ax.set_ylim([min(spatial_result_1._points_y), max(spatial_result_1._points_y)])

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
