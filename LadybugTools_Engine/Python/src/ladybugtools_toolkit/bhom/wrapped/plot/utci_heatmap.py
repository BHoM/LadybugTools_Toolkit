"""Method to wrap UTCI plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path
from unittest.util import _MIN_COMMON_LEN
import matplotlib

def utci_heatmap(json_file:str,
            return_file: str,
            save_path = None) -> None:
    try:
        from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
        from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
        import matplotlib.pyplot as plt
        import numpy as np
        import json
    
        with open(json_file, "r") as args:
            argsDict = json.loads(args.read())
    
        ec = ExternalComfort.from_dict(json.loads(argsDict["external_comfort"]))

        custom_bins = UTCI_DEFAULT_CATEGORIES

        bin_colours = json.loads(argsDict["bin_colours"])

        if len(bin_colours) == 10:
            custom_bins = Categorical(
                bins=(-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf),
                colors=(bin_colours),
                name="UTCI")

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ec.plot_utci_heatmap(utci_categories = custom_bins)

        utci_collection = ec.universal_thermal_climate_index

        return_dict = {"data": utci_metadata(utci_collection), "external_comfort": ec.to_dict()}

        plt.tight_layout()
    
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path
    
        with open(return_file, "w") as rtn:
            rtn.write(json.dumps(return_dict, default=str))
    
        print(return_file)
    except Exception as ex:
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap"
        )
    )
    parser.add_argument(
        "-in",
        "--json_args",
        help="helptext",
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
        "-sp",
        "--save_path",
        help="helptext",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    matplotlib.use("Agg")
    utci_heatmap(args.json_args, args.return_file, args.save_path)