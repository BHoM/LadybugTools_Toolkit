import argparse
import matplotlib
import traceback

def walkability_heatmap(json_file: str, return_file: str, save_path: str):
    try:
        from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
        from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import json
        import matplotlib.pyplot as plt

        with open(json_file, "r") as args:
            argsDict = json.loads(args.read())
    
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
    
        with open(return_file, "w") as rtn:
            rtn.write(json.dumps(return_dict, default=str))
    
        print(return_file)

    except Exception as ex:
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an external comfort object, extract a walkability heatmap"
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
    walkability_heatmap(args.json_args, args.return_file, args.save_path)