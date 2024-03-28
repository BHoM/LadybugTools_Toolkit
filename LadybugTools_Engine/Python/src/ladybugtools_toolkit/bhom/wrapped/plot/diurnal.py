"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path

def diurnal(epw_file, data_type_key="Dry Bulb Temperature", color="#000000", title=None, period="monthly", save_path = None):
    try:
        from ladybug.epw import EPW, AnalysisPeriod
        from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
        from ladybugtools_toolkit.plot._diurnal import diurnal
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt
    
        epw = EPW(epw_file)
        data_type_key = data_type_key.replace("_"," ")
        coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])
        fig = diurnal(collection_to_series(coll),title=title, period=period, color=color).get_figure()
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
        "-dtk",
        "--data_type_key",
        help="Key in EPW data to create a plot from.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--colour",
        help="Colour of the line",
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
    diurnal(args.epw_file, args.data_type_key, args.colour, args.title, args.period, args.save_path)
