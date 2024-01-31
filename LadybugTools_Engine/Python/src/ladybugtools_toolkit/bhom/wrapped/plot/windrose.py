"""Method to wrap for creating wind roses from epw files."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def main(epw_file: str, month_from:int = 1, day_from:int = 1, hour_from:int = 0, month_to:int = 12, day_to:int = 31, hour_to:int = 23, colour_map: str = "YlGnBu", bins:int = 36, save_path:str = None) -> None:
    """Method to wrap for creating wind roses from epw files."""
    try:
        from ladybug.epw import EPW, AnalysisPeriod
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.wind import Wind
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt
        from pathlib import Path

        epw = EPW(epw_file)
        analysis_period = AnalysisPeriod(month_from,day_from,hour_from,month_to,day_to,hour_to)
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
        "-cmap",
        "--colour_map",
        help="Matplotlib colour map to use.",
        type=str,
        required=False,
        )
    parser.add_argument(
        "-p",
        "--save_path",
        help="Path where to save the output image.",
        type=str,
        required=False,
        )

    parser.add_argument(
        "-m0",
        "--month_from",
        help="Analysis period",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-d0",
        "--day_from",
        help="Analysis period",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-h0",
        "--hour_from",
        help="Analysis period",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-m1",
        "--month_to",
        help="Analysis period",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-d1",
        "--day_to",
        help="Analysis period",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-h1",
        "--hour_to",
        help="Analysis period",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-bins",
        "--bins",
        help="Number of bins",
        type=int,
        required=False,
    )

    args = parser.parse_args()
    main(args.epw_file, args.month_from, args.day_from, args.hour_from, args.month_to, args.day_to, args.hour_to, args.colour_map, args.bins, args.save_path)