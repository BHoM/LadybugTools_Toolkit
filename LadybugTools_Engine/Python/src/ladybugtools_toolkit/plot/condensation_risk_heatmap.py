"""Plotting methods for condensation risk."""

import argparse
from pathlib import Path
import matplotlib
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


def condensation_risk_heatmap(epw_file: str, thresholds: list[float], return_file: str, save_path: str = None) -> None:
    """Create a heatmap of the condensation potential for a given set of
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
    """
    epw = EPW(epw_file)
    dbt_series = collection_to_series(epw.dry_bulb_temperature)

    header = header_from_string("Condensation Thresholds (C)")
    hcc = HourlyContinuousCollection(header = header, values = dbt_series.values)
    
    thresholds = [-20, -15, -10, -5, 0]
    thresholds.insert(0,-np.inf)
    thresholds.append(np.inf)

    CATEGORIES = condensation_categories_from_thresholds(thresholds)

    fig =   CATEGORIES.annual_heatmap(
            series=collection_to_series(epw.dew_point_temperature),
            ax=None,
            title="Condensation Risk Heatmap",
        ).get_figure()

    return_dict = {"data": collection_metadata(hcc)}

    if save_path == None or save_path == "":
        base64 = figure_to_base64(fig,html=False)
        return_dict["figure"] = base64
    else:
        fig.savefig(save_path, dpi=150, transparent=True)
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
    condensation_risk_heatmap(args.epw_file, args.return_file, args.save_path)