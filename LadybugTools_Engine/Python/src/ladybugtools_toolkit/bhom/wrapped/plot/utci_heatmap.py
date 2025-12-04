"""Method to wrap UTCI plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import sys
import traceback
import matplotlib
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
import matplotlib.pyplot as plt
import numpy as np
import json
from ...logger import CONSOLE_LOGGER

PARSER = argparse.ArgumentParser(
    description=(
        "Given an EPW file path, extract a heatmap"
    )
)
PARSER.add_argument(
    "-in",
    "--input_json",
    help="helptext",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-sp",
    "--save_path",
    help="helptext",
    type=str,
    required=False,
)

def utci_heatmap(input_json:str, save_path = None) -> str:
    try:

        if not input_json.startswith("{"): #assume it's a path
            with open(input_json, "r") as f:
                input_json = f.read()

        argsDict = json.loads(input_json)
    
        ec = ExternalComfort.from_dict(json.loads(argsDict["external_comfort"]))

        custom_bins = UTCI_DEFAULT_CATEGORIES

        bin_colours = json.loads(argsDict["bin_colours"])

        if len(bin_colours) == 10:
            custom_bins = Categorical(
                bins=(-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf),
                colors=(bin_colours),
                name="UTCI")

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ec.plot_utci_heatmap(utci_categories = custom_bins, ax=ax)

        utci_collection = ec.universal_thermal_climate_index

        return_dict = {"data": utci_metadata(utci_collection), "external_comfort": ec.to_dict()}

        plt.tight_layout()
    
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path
    
        plt.close(fig)

        return json.dumps(return_dict, default=str)
    except Exception:
        CONSOLE_LOGGER.error("UTCI Heatmap could not be created.", exc_info=1)
        return ""

if __name__ == "__main__":
    args = PARSER.parse_args()
    matplotlib.use("Agg")
    utci_heatmap(args.input_json, args.save_path)