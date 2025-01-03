﻿"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import json
import traceback
from pathlib import Path
import matplotlib

def diurnal(epw_file, return_file: str, data_type_key="Dry Bulb Temperature", color="#000000", title=None, period="monthly", save_path = None):
    try:
        from ladybug.epw import EPW, AnalysisPeriod
        from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
        from ladybugtools_toolkit.ladybug_extension.epw import wet_bulb_temperature
        from python_toolkit.plot.diurnal import diurnal
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
        import matplotlib.pyplot as plt
        
        epw = EPW(epw_file)
        
        if data_type_key == "Wet Bulb Temperature":
            coll = wet_bulb_temperature(epw)
        else:
            coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])
        
        fig = diurnal(collection_to_series(coll),title=title, period=period, color=color).get_figure()
        return_dict = {"data": collection_metadata(coll)}
        
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path

        with open(return_file, "w") as rtn:
            rtn.write(json.dumps(return_dict, default=str))
           
        print(return_file)

    except Exception as e:
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a diurnal plot"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_file",
        help="The EPW file to extract a diurnal plot from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-dtk",
        "--data_type_key",
        help="Key in EPW data to create a plot from.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--colour",
        help="Colour of the line",
        type=str,
        required=True,
        )
    parser.add_argument(
        "-t",
        "--title",
        help="Title that the plot will have",
        type=str,
        required=True,
        )
    parser.add_argument(
        "-ap",
        "--period",
        help="Period that will be plotted on the diurnal plot",
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
    diurnal(args.epw_file, args.return_file, args.data_type_key, args.colour, args.title, args.period, args.save_path)
