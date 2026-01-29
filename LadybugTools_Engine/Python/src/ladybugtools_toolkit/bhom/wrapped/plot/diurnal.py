"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import json
import sys
import traceback
from pathlib import Path
import matplotlib
from ladybug.epw import EPW, AnalysisPeriod
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.ladybug_extension.epw import wet_bulb_temperature
from python_toolkit.plot.diurnal import diurnal as dnal
from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER

PARSER = argparse.ArgumentParser(
    description=(
        "Given an EPW file path, extract a diurnal plot"
    )
)
PARSER.add_argument(
    "-e",
    "--epw_file",
    help="The EPW file to extract a diurnal plot from",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-dtk",
    "--data_type_key",
    help="Key in EPW data to create a plot from.",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-colour",
    "--colour",
    help="Colour of the line",
    type=str,
    required=True,
    )
PARSER.add_argument(
    "-t",
    "--title",
    help="Title that the plot will have",
    type=str,
    required=True,
    )
PARSER.add_argument(
    "-ap",
    "--period",
    help="Period that will be plotted on the diurnal plot",
    type=str,
    required=True,
    )
PARSER.add_argument(
    "-p",
    "--save_path",
    help="Path where to save the output image.",
    type=str,
    required=False,
    )

def diurnal(epw_file, data_type_key="Dry Bulb Temperature", colour="#000000", title=None, period="monthly", save_path = None) -> str:
    try:
        epw = EPW(epw_file)
        
        if data_type_key == "Wet Bulb Temperature":
            coll = wet_bulb_temperature(epw)
        else:
            coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])
        
        fig, ax = plt.subplots()

        dnal(collection_to_series(coll), ax=ax, title=title, period=period, color=colour)
        return_dict = {"data": collection_metadata(coll)}
        
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path

        plt.close(fig)

        return json.dumps(return_dict, default=str)

    except Exception:
        CONSOLE_LOGGER.error("Diurnal plot could not be created.", exc_info=1)
        return traceback.format_exc()

if __name__ == "__main__":
    args = PARSER.parse_args()
    matplotlib.use("Agg")
    diurnal(args.epw_file, args.return_file, args.data_type_key, args.colour, args.title, args.period, args.save_path)
