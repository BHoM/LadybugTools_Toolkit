"""Plotting methods for condensation risk."""

import argparse
import textwrap

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import numpy as np
from ladybug.epw import EPW
from python_toolkit.plot.heatmap import heatmap
from matplotlib.colors import LinearSegmentedColormap
from ladybugtools_toolkit.ladybug_extension.header import header_from_string
from ladybug.epw import AnalysisPeriod, HourlyContinuousCollection
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.categorical.categories import UTCI_DEFAULT_CATEGORIES, Categorical

def condensation_categories_from_thresholds(thresholds: tuple[float]) -> Categorical:
    """Create a categorical from provided threshold temperatures.

    Args:
        thresholds (tuple[float]):
            The temperature thresholds to be used.

    Returns:
        Categorical: The resulting categorical object with condensation risk colouring.
    """
    cmap = LinearSegmentedColormap.from_list("condensation", ["indigo", "royalblue", "white"], N=100)
    return Categorical.from_cmap(thresholds, cmap)

def facade_condensation_risk_chart(epw_file: str, thresholds: list[float], **kwargs) -> Figure:
    """Create a chart with thresholds of the condensation potential for a given set of
    timeseries dry bulb temperatures from an EPW.

    Args:
        epw_file (string):
            The input EPW file.
        thresholds (list[float]):
            The temperature thresholds to use.
        return_file (string):
            The filepath to write the resulting JSON to.
        save_path (string):
            The filepath to save the resulting image file of the heatmap to.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        Figure: A matplotlib Figure object.
    """
    epw = EPW(epw_file)
    series = collection_to_series(epw.dry_bulb_temperature)
    
    thresholds.insert(0,-np.inf)
    thresholds.append(np.inf)

    CATEGORIES = condensation_categories_from_thresholds(thresholds)

    title = kwargs.pop("title", None)
    figsize = kwargs.pop("figsize", (15, 8))

    # Instantiate figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[6, 5], hspace=0)
    chart_ax = fig.add_subplot(spec[0, 0])
    table_ax = fig.add_subplot(spec[1, 0])

    # Add Thresholds Chart
    CATEGORIES.annual_threshold_chart(series, chart_ax, color = 'slategrey', **kwargs)
   
    # Add table
    CATEGORIES.annual_monthly_table(series, table_ax, False, True, **kwargs)

    title = f"{title}" if title is not None else series.name
    chart_ax.set_title(title, y=1, ha="left", va="bottom", x=0)
    chart_ax.set_anchor('W')

    return fig


def facade_condensation_risk_heatmap_histogram(epw_file: str, thresholds: list[float], **kwargs) -> Figure:
    """Create a histogram of the condensation potential for a given set of
    timeseries dry bulb temperatures from an EPW.

    Args:
        epw_file (string):
            The input EPW file.
        thresholds (list[float]):
            The temperature thresholds to use.
        return_file (string):
            The filepath to write the resulting JSON to.
        save_path (string):
            The filepath to save the resulting image file of the heatmap to.
        **kwargs:
            Additional keyword arguments to pass to the heatmap function.

    Returns:
        Figure: A matplotlib Figure object.
    """
    epw = EPW(epw_file)
    series = collection_to_series(epw.dry_bulb_temperature)
    
    thresholds.insert(0,-np.inf)
    thresholds.append(np.inf)

    CATEGORIES = condensation_categories_from_thresholds(thresholds)

    title = kwargs.pop("title", None)
    figsize = kwargs.pop("figsize", (15, 8))

    # Instantiate figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[5, 3], hspace=0.0)
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])

    # Add heatmap
    CATEGORIES.annual_heatmap(series, heatmap_ax, **kwargs)

    # Add stacked plot
    CATEGORIES.annual_monthly_histogram(series, histogram_ax, False, True, **kwargs)

    title = f"{series.name} - {title}" if title is not None else series.name
    heatmap_ax.set_title(title, y=1, ha="left", va="bottom", x=0)

    return fig