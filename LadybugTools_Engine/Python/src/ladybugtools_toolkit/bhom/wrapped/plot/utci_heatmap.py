"""Method to wrap UTCI plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def main(epw_file:str, 
            ground_material,#:Material, 
            shade_material,#:Material, 
            typology,#:Typology, 
            evaporative_cooling = 0, 
            wind_speed_multiplier:float = None,
            bin_colours = None,#:List<color>,
            save_path = None) -> None:
    try:
        from ladybugtools_toolkit.external_comfort.material import Materials
        from ladybugtools_toolkit.external_comfort.typology import Typologies
        from ladybugtools_toolkit.external_comfort.simulate import SimulationResult
        from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np

        sr = SimulationResult(epw_file, ground_material, shade_material)
        typology.evaporative_cooling = [evaporative_cooling]*8760
        typology.wind_speed_multiplier = wind_speed_multiplier
        ec = ExternalComfort(sr, typology)

        custom_bins = UTCI_DEFAULT_CATEGORIES

        if bin_colours is not None:
            custom_bins = Categorical(
                bins=(-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf),
                colors=(bin_colours),
                name="UTCI")

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ec.plot_utci_heatmap(utci_categories = custom_bins)

        plt.tight_layout()
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            print(base64)
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            print(save_path)
        plt.close(fig)

    except Exception as e:
        print(e)



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
        "-gm",
        "--ground_material",
        help="helptext",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-sm",
        "--shade_material",
        help="helptext",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--typology",
        help="helptext",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ec",
        "--evaporative_cooling",
        help="helptext",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-ws",
        "--wind_speed_multiplier",
        help="helptext",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-bc",
        "--bin_colours",
        help="helptext",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-sp",
        "--save_path",
        help="helptext",
        type=str,
        required=False,
    )


    args = parser.parse_args()
    main(args.epw_path, args.ground_material, args.shade_material, args.typology, args.evaporative_cooling, args.wind_speed_multiplier, args.bin_colours, args.save_path)