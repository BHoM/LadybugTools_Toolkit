"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
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
from ladybugtools_toolkit.plot.condensation_risk_heatmap import *


def condensation_risk_heatmap(epw_file: str, thresholds: list[float], return_file: str, save_path: str = None, **kwargs) -> Figure:
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
    hcc = epw.dry_bulb_temperature

    fig = condensation_risk_heatmap_histogram(epw_file, thresholds, return_file, save_path).get_figure()

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
