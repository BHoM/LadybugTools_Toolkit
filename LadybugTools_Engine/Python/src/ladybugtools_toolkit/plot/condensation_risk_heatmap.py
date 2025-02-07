"""Plotting methods for condensation risk."""

import argparse
from pathlib import Path
import matplotlib
import json
from ladybug.epw import EPW
from python_toolkit.plot.heatmap import heatmap
from matplotlib.colors import LinearSegmentedColormap
from ladybugtools_toolkit.ladybug_extension.header import header_from_string
from ladybug.epw import AnalysisPeriod, HourlyContinuousCollection
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata

thresholds = [
    (0, 4.4),
    (1,1.7),
    (2,0),
    (3,-1.1),
    (4,-3.9),
    (5,-6.7),
    (6,-9.4),
    (7,-12.2),
    (8,-13.5),
    (9,-15),
    (10,-17.8),
    (11, -999)
]

def get_threshold(dbt):
    matches = [a for a in thresholds if a[1] <= dbt]
    return min([a[0] for a in matches])

def dbt_to_condensation_risk(dbt_series):
    con_risk = dbt_series.apply(get_threshold)
    header = header_from_string("Condensation risk (Index)")
    hcc = HourlyContinuousCollection(header = header, values = con_risk.values)
    return hcc

def condensation_risk_heatmap(epw_file: str, return_file: str, save_path: str = None) -> None:
    epw = EPW(epw_file)
    dbt_series = collection_to_series(epw.dry_bulb_temperature)
    con_risk = dbt_to_condensation_risk(dbt_series)
    cmap = LinearSegmentedColormap.from_list("condensation", ["white","blue","purple","black"], N=100)
    fig = heatmap(collection_to_series(con_risk), vmin=0, vmax=11, cmap=cmap, ).get_figure()

    return_dict = {"data": collection_metadata(con_risk)}

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