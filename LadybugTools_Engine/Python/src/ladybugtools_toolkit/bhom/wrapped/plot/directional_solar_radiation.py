"""Method to wrap creation of panel orientation plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path
import os
import matplotlib
from ladybugtools_toolkit.solar import IrradianceType, tilt_orientation_factor, create_radiation_matrix
from ladybug.wea import AnalysisPeriod
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.bhom.wrapped.metadata.solar_radiation_metadata import solar_radiation_metadata
import matplotlib.pyplot as plt
from pathlib import Path
import json
from ...logger import CONSOLE_LOGGER

PARSER = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap"
        )
    )
PARSER.add_argument(
    "-e",
    "--epw_file",
    help="The EPW file to extract a heatmap from",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-d",
    "--directions",
    help="The number of directions to use when plotting orientations.",
    type=int,
    required=True,
)
PARSER.add_argument(
    "-ti",
    "--tilts",
    help="The number of tilts to use when plotting orientations.",
    type=int,
    required=True,
)
PARSER.add_argument(
    "-ir",
    "--irradiance_type",
    help="The irradiance type to use.",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-cmap",
    "--cmap",
    help="Matplotlib colour map to use.",
    type=str,
    required=True,
    )
PARSER.add_argument(
    "-ap",
    "--analysis_period",
    help="Analysis period",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-t",
    "--title",
    help="The title to be displayed on the plot.",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-p",
    "--save_path",
    help="Path to save the output image.",
    type=str,
    required=False,
    )

def directional_solar_radiation(epw_file, directions, tilts, irradiance_type, analysis_period, cmap, title, save_path) -> str:
    try:
        if cmap not in plt.colormaps():
            cmap = "YlOrRd"

        analysis_period = AnalysisPeriod.from_dict(json.loads(analysis_period))

        if irradiance_type == "Total":
            irradiance_type = IrradianceType.TOTAL
        elif irradiance_type == "Diffuse":
            irradiance_type = IrradianceType.DIFFUSE
        elif irradiance_type == "Direct":
            irradiance_type = IrradianceType.DIRECT
        elif irradiance_type == "Reflected":
            irradiance_type = IrradianceType.REFLECTED

        fig, ax = plt.subplots(1, 1, figsize=(22.8/2, 7.6/2))
        values, dirs, tts = create_radiation_matrix(Path(epw_file), rad_type=irradiance_type, analysis_period=analysis_period, directions=directions, tilts=tilts)
        tilt_orientation_factor(Path(epw_file), ax=ax, rad_type=irradiance_type, analysis_period=analysis_period, directions=directions, tilts=tilts, cmap=cmap)
        if not (title == "" or title is None):
            ax.set_title(title)

        return_dict = {}

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path
        
        return_dict["data"] = solar_radiation_metadata(values, dirs, tts)

        plt.close(fig)

        return json.dumps(return_dict, default=str)
    
    except Exception:
        CONSOLE_LOGGER.error("Solar Radiation plot could not be created.", exc_info=1)
        return ""

if __name__ == "__main__":
    args = PARSER.parse_args()

    os.environ["TQDM_DISABLE"] = "1" # set an environment variable so that progress bars are disabled for the simulation process
    matplotlib.use("Agg")
    directional_solar_radiation(args.epw_file, args.directions, args.tilts, args.irradiance_type, args.analysis_period, args.colour_map, args.title, args.save_path)
    del os.environ["TQDM_DISABLE"] # unset the env variable