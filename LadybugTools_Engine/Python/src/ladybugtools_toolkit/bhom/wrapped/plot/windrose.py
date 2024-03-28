"""Method to wrap for creating wind roses from epw files."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def windrose(epw_file: str, analysis_period: str, colour_map: str, bins: int, save_path: str = None) -> None:
    """Method to wrap for creating wind roses from epw files."""
    try:
        from ladybug.epw import EPW, AnalysisPeriod
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.wind import Wind
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt
        from pathlib import Path
        import json

        if colour_map not in plt.colormaps():
            colour_map = "YlGnBu"

        epw = EPW(epw_file)
        analysis_period = AnalysisPeriod.from_dict(json.loads(analysis_period))
        w_epw = Wind.from_epw(epw_file)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})
        w_epw.filter_by_analysis_period(analysis_period=analysis_period).plot_windrose(ax=ax, directions=bins, ylim=(0, 3.6/bins), colors=colour_map)

        plt.tight_layout()
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            print(base64)
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            print(save_path)
            
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
        "--epw_file",
        help="The EPW file to extract a heatmap from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ap",
        "--analysis_period",
        help="Analysis period",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-cmap",
        "--colour_map",
        help="Matplotlib colour map to use.",
        type=str,
        required=True,
        )
    parser.add_argument(
        "-bins",
        "--bins",
        help="Number of bins",
        type=int,
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
    windrose(args.epw_file, args.analysis_period, args.colour_map, args.bins, args.save_path)