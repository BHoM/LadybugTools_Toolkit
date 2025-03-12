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
from ..categorical.categories import UTCI_DEFAULT_CATEGORIES, Categorical

thresholds = [-20, 
              -15, 
              -10, 
              -5, 
              0]

def condensation_categories_from_thresholds(
    thresholds: tuple[float],
):
    """Create a categorical from provided threshold temperatures.

    Args:
        thresholds (tuple[float]):
            The temperature thresholds to be used.

    Returns:
        Categories: The resulting categories object.
    """
    cmap = LinearSegmentedColormap.from_list("condensation", ["black","purple","blue","white"], N=100)
    return Categorical.from_cmap(thresholds, cmap)

def condensation_risk_heatmap_histogram(epw_file: str, thresholds: list[float], return_file: str, save_path: str = None, **kwargs) -> None:
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
    """
    epw = EPW(epw_file)
    series = collection_to_series(epw.dry_bulb_temperature)

    hcc = epw.dry_bulb_temperature
    
    thresholds.insert(0,-np.inf)
    thresholds.append(np.inf)

    CATEGORIES = condensation_categories_from_thresholds(thresholds)

    title = kwargs.pop("title", None)
    figsize = kwargs.pop("figsize", (15, 10))

    # Instantiate figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[5, 2, 5], hspace=0.0)
    heatmap_ax = fig.add_subplot(spec[0, 0])
    histogram_ax = fig.add_subplot(spec[1, 0])
    chart_ax = fig.add_subplot(spec[2, 0])

    # Add heatmap
    CATEGORIES.annual_heatmap(series, heatmap_ax)

    # Add stacked plot
    CATEGORIES.annual_monthly_histogram(series, histogram_ax, False, True)

    # Add Thresholds Chart
    CATEGORIES.annual_threshold_chart(series, chart_ax, color = 'slategrey')

    title = f"{series.name} - {title}" if title is not None else series.name
    heatmap_ax.set_title(title, y=1, ha="left", va="bottom", x=0)

    return_dict = {"data": hcc}

    if save_path == None or save_path == "":
        base64 = figure_to_base64(fig,html=False)
        return_dict["figure"] = base64
    else:
        fig.savefig(save_path, dpi=300, transparent=True)
        return_dict["figure"] = save_path
    
    with open(return_file, "w") as rtn:
        rtn.write(json.dumps(return_dict, default=str))
    
    print(return_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap of condensation risk"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_file",
        help="The EPW file to extract a heatmap from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--thresholds",
        help="thresholds to use.",
        type=list[float],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--return_file",
        help="json file to write return data to.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--save_path",
        help="Path where to save the output image.",
        type=str,
        required=False,
        )

    args = parser.parse_args()
    matplotlib.use("Agg")
    condensation_risk_heatmap_histogram(args.epw_file, args.thresholds, args.return_file, args.save_path)