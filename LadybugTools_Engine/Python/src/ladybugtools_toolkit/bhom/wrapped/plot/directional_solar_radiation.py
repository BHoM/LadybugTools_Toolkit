"""Method to wrap creation of panel orientation plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path
import os

def directional_solar_radiation(epw_file, azimuths, altitudes, ground_reflectance, irradiance_type, isotropic, analysis_period, title, save_path):
    try:
        from ladybugtools_toolkit.solar import Solar, IrradianceType
        from ladybug.wea import EPW, AnalysisPeriod
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt
        from pathlib import Path
        import json

        analysis_period = AnalysisPeriod.from_dict(json.loads(analysis_period))

        if irradiance_type == "Total":
            irradiance_type = IrradianceType.TOTAL
        elif irradiance_type == "Diffuse":
            irradiance_type = IrradianceType.DIFFUSE
        elif irradiance_type == "Direct":
            irradiance_type = IrradianceType.DIRECT
        elif irradiance_type == "Reflected":
            irradiance_type = IrradianceType.REFLECTED

        epw = EPW(epw_file)
        solar = Solar.from_epw(epw)
        fig, ax = plt.subplots(1, 1, figsize=(22.8/2, 7.6/2))
        solar.plot_tilt_orientation_factor(location=epw.location, ax=ax, azimuths=azimuths, altitudes=altitudes, isotropic=isotropic, irradiance_type=irradiance_type, analysis_period=analysis_period, ground_reflectance=ground_reflectance).get_figure()
        ax.set_title(title)


        if save_path is None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            print(base64)
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            print(save_path)
    except Exception as ex:
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a heatmap"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_file",
        help="The EPW file to extract a heatmap from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-az",
        "--azimuths",
        help="The azimuths to use when plotting orientations",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-al",
        "--altitudes",
        help="The altitudes to use when plotting orientations",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-gr",
        "--ground_reflectance",
        help="The ground reflectance, between 0 and 1",
        type=float,
        required=True,
    )
    parser.add_argument(
        "-ir",
        "--irradiance_type",
        help="The irradiance type to plot",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-iso",
        "--isotropic",
        help="Whether the method should calculate isotropic diffuse irradiance",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-ap",
        "--analysis_period",
        help="Analysis period",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="The title displayed on the plot",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--save_path",
        help="Path where to save the output image.",
        type=str,
        required=False,
        )

    args = parser.parse_args()

    os.environ["TQDM_DISABLE"] = "1" # set an environment variable so that progress bars are disabled for the simulation process
    directional_solar_radiation(args.epw_file, args.azimuths, args.altitudes, args.ground_reflectance, args.irradiance_type, args.isotropic, args.analysis_period, args.title, args.save_path)
    del os.environ["TQDM_DISABLE"] # unset the env variable