import argparse
import matplotlib

def comfort_simulation(epw_file:str,
            json_file:str,
            return_file: str,
            wind_speed_multiplier:float = 1) -> None:
    from ladybugtools_toolkit.external_comfort.material import Materials
    from ladybugtools_toolkit.external_comfort.typology import Typologies
    from ladybugtools_toolkit.external_comfort._typologybase import Typology
    from ladybugtools_toolkit.external_comfort.simulate import SimulationResult
    from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
    from ladybugtools_toolkit.bhom.to_bhom import hourlycontinuouscollection_to_bhom
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

    return_dict = {"mrt": hourlycontinuouscollection_to_bhom(ec.mean_radiant_temperature), "utci": hourlycontinuouscollection_to_bhom(ec.universal_thermal_climate_index)}

    with open(return_file, "w") as rtn:
        rtn.write(json.dumps(return_dict))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_path",
        help="path to weather file to use in simulation.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-in",
        "--json_args",
        help="JSON file input containing necessary information to run the simulation.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ws",
        "--wind_speed_multiplier",
        help="wind speed multiplier to apply to epw file before simulation.",
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

    args = parser.parse_args()
    matplotlib.use("Agg")
    comfort_simulation(args.epw_path, args.json_args, args.return_file, args.wind_speed_multiplier)