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
    (0,0),
    (1,3),
    (2,6),
    (3,12),
    (4,15),
    (5,18),
    (6,21),
    (7,24),
    (8,27),
    (9,30),
    (10,33),
    (11,999)
]

def get_threshold(dbt_delta):
    matches = [a for a in thresholds if a[1] >= dbt_delta]
    return min([a[0] for a in matches])

def dbt_to_condensation_risk(dbt_series, internal_rh, internal_temp):
    """Calculate condensation risk based on external temperature and internal dew point, using Mark G. Lawrence's DPT approximation for RH>50%.

    Args:
        dbt_series (list[float]):
            List of Dry Bulb Temperatures
        internal_rh (int):
            Internal Relative Humidity, as a percentage
        internal_temp (float):
            Internal Temperature of the building
    """
    dew_point_temp = internal_temp-((100-internal_rh)/5)
    dbt_delta = dew_point_temp - dbt_series
    con_risk = dbt_delta.apply(get_threshold)
    header = header_from_string("Condensation risk (Index)")
    hcc = HourlyContinuousCollection(header = header, values = con_risk.values)
    return hcc

def condensation_risk_heatmap(epw_file: str, return_file: str, save_path: str = None) -> None:
    """Create a heatmap of the condensation potential for a given set of
    timeseries dry bulb temperatures from an EPW.

    Args:
        epw_file (string):
            The input EPW file.
        return_file (string):
            The filepath to write the resulting JSON to.
        save_path (string):
            The filepath to save the resulting image file of the heatmap to.
    """
    epw = EPW(epw_file)
    internal_temp = 21 #default value for internal temp
    internal_rh = 40 #default value for internal rh
    dbt_series = collection_to_series(epw.dry_bulb_temperature)
    dpt_series = collection_to_series(epw.dew_point_temperature)
    con_risk = dbt_to_condensation_risk(dbt_series, internal_rh, internal_temp)
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