"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import os
import json
import sys
import traceback
import matplotlib
import matplotlib.figure
from ladybug.epw import EPW
from ladybugtools_toolkit.plot.compare import compare_epw_key_line, compare_epw_key_hist
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
    action="extend",
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
PARSER.add_argument(
    "-l",
    "--line",
    help="Produce a line plot instead of a histogram",
    action="store_true",
    default=False
    )

def epw_comparison(epw_file: str, epw_list: List[str], data_type_key: str, line:bool, save_path:str = None) -> str:
    """Create a timeseries plot with a line for each epw file for the specified data key and return it in a format readable by the LadybugToolsAdapter."""
    try:
        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")

        with plt.style.context(style):
            fig, ax = plt.subplots()

            epws = [EPW(epw_file)]
            epws.extend([EPW(f) for f in epw_list])
            
            if line:
                compare_epw_key_line(epws, key=data_type_key.lower().strip().replace(" ", "_"), style_context=style, ax=ax)
            else:
                compare_epw_key_hist(epws, key=data_type_key.lower().strip().replace(" ", "_"), style_context=style, ax=ax)

        return_dict = {}

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path

        plt.close(fig)

        return_dict["data"] = None #Unsure of how to create representative collection metadata for a comparison plot type that doesn't simply list every epw file compared

        return json.dumps(return_dict, default=str)
            
    except Exception:
        CONSOLE_LOGGER.error("Timeseries comparison could not be created.", exc_info=1)
        return traceback.format_exc()

if __name__ == "__main__":

    args = PARSER.parse_args()
    matplotlib.use("Agg")

    epw_comparison(args.epw_file, args.epw_list, args.data_type_key, args.line, args.save_path)
