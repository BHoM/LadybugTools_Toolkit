"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import os
from pathlib import Path
import json
import sys
import traceback
import matplotlib
import matplotlib.figure
from ladybug.epw import EPW
from ladybug.datacollection import HourlyContinuousCollection
from python_toolkit.plot.heatmap import heatmap as hmap
from ladybugtools_toolkit.plot.compare import compare_epw_key_line
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
from ladybugtools_toolkit.ladybug_extension.epw import wet_bulb_temperature
from ladybugtools_toolkit.plot.utilities import figure_to_base64
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER
from typing import List

PARSER = argparse.ArgumentParser(
    description=(
        "Given an EPW file path, and a list of epws to compare to, construct a line chart for a specific epw key."
    )
)
PARSER.add_argument(
    "-e",
    "--epw_file",
    help="The EPW file to compare from",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-el",
    "--epw_list",
    help="List of EPW files to compare with the base",
    type=str,
    nargs='*',
    required=True,
)
PARSER.add_argument(
    "-dtk",
    "--data_type_key",
    help="Key to compare.",
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

def epw_comparison(epw_file: str, epw_list: List[str], data_type_key: str, save_path:str = None) -> str:
    """Create a CSV file version of an EPW."""
    try:
        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")
        if colour_map not in plt.colormaps():
            colour_map = "YlGnBu"

        with plt.style.context(style):
            fig, ax = plt.subplots()

            epw_list = [EPW(epw_file)]
            epw_list.extend([EPW(f) for f in epw_list])
        
            coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])
        
            compare_epw_key_line(epw_list, key=data_type_key.lower().replace(" ", "_"), style_context=style, ax=ax)

        return_dict = {}

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path

        plt.close(fig)

        return_dict["data"] = collection_metadata(coll)

        return json.dumps(return_dict, default=str)
            
    except Exception:
        CONSOLE_LOGGER.error("Heatmap could not be created.", exc_info=1)
        return traceback.format_exc()


if __name__ == "__main__":

    args = PARSER.parse_args()
    matplotlib.use("Agg")
    epw_comparison(args.epw_file, args.epw_list, args.data_type_key, args.save_path)
