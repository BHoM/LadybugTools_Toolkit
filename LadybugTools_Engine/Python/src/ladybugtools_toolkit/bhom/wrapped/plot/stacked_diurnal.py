"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path

def stacked_diurnal(epw_file, data_type_keys = ["Dry Bulb Temperature"], colours=["#000000"], title=None, period="monthly", save_path = None):
    try:
        from ladybug.epw import EPW, AnalysisPeriod
        from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
        from ladybugtools_toolkit.plot._diurnal import stacked_diurnals
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt
    
        epw = EPW(epw_file)
        data_type_keys = [k.replace("_", " ") for k in data_type_keys]
        epw_dict = epw.to_dict()

        colls = []
        for key in data_type_keys:
            colls.append(collection_to_series(HourlyContinuousCollection.from_dict([coll for coll in epw_dict["data_collections"] if coll["header"]["data_type"]["name"] == key][0])))

        fig = stacked_diurnals(colls, title=title, period=period, colors=colours).get_figure()
        
        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            print(base64)
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            print(save_path)
    except Exception as e:
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, extract a diurnal plot"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_file",
        help="The EPW file to extract a diurnal plot from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-dtks",
        "--data_type_keys",
        help="Key in EPW data to create a plot from.",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--colours",
        help="Colour of the line",
        nargs="+",
        type=str,
        required=True,
        )
    parser.add_argument(
        "-t",
        "--title",
        help="Title that the plot will have",
        type=str,
        required=True,
        )
    parser.add_argument(
        "-ap",
        "--period",
        help="Period that will be plotted on the diurnal plot",
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
    stacked_diurnal(args.epw_file, args.data_type_keys, args.colours, args.title, args.period, args.save_path)
