import argparse
import matplotlib
import traceback
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
from ladybugtools_toolkit.plot.utilities import figure_to_base64
import json
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER

PARSER = argparse.ArgumentParser(
    description=(
        "Given an external comfort object, extract a walkability heatmap"
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

def walkability_heatmap(input_json: str, save_path: str) -> str:
    try:
        argsDict = json.loads(input_json)
    
        ec = ExternalComfort.from_dict(json.loads(argsDict["external_comfort"]))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ec.plot_walkability_heatmap(ax=ax)

        #TODO: create walkability collection metadata
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
        CONSOLE_LOGGER.error("Walkability plot could not be created.", exc_info=1)
        return ""

if __name__ == "__main__":

    args = PARSER.parse_args()
    matplotlib.use("Agg")
    walkability_heatmap(args.json_args, args.save_path)