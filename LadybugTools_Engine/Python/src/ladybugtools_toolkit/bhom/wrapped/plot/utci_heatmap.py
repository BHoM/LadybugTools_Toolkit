"""Method to wrap UTCI plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path
from unittest.util import _MIN_COMMON_LEN
import matplotlib

def utci_heatmap(epw_file:str,
            json_file:str,
            return_file: str,
            wind_speed_multiplier:float = 1,
            save_path = None) -> None:
    from ladybugtools_toolkit.external_comfort.material import Materials
    from ladybugtools_toolkit.external_comfort.typology import Typologies
    from ladybugtools_toolkit.external_comfort._typologybase import Typology
    from ladybugtools_toolkit.external_comfort.simulate import SimulationResult
    from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
    from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
    from ladybugtools_toolkit.plot.utilities import figure_to_base64
    from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
    from honeybee_energy.dictutil import dict_to_material
    from ladybug.epw import EPW
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import json
    
    with open(json_file, "r") as args:
        argsDict = json.loads(args.read())
    
    typology = Typology.from_dict(json.loads(argsDict["typology"]))
    ground_material = dict_to_material(json.loads(argsDict["ground_material"]))
    shade_material = dict_to_material(json.loads(argsDict["shade_material"]))

    sr = SimulationResult(epw_file, ground_material, shade_material)
    epw = EPW(epw_file)
    typology.target_wind_speed = epw.wind_speed * wind_speed_multiplier
    ec = ExternalComfort(sr, typology)

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

    return_dict = {"data": utci_metadata(utci_collection)}

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_path",
        help="helptext",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-in",
        "--json_args",
        help="helptext",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ws",
        "--wind_speed_multiplier",
        help="helptext",
        type=float,
        required=False,
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
    utci_heatmap(args.epw_path, args.json_args, args.return_file, args.wind_speed_multiplier, args.save_path)