"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def main(epw_file: str, data_type_key: str, colour_map: str, save_path:str = None) -> None:
    """Create a CSV file version of an EPW."""
    try:
        from ladybug.epw import EPW
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.plot._heatmap import heatmap
        from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
        from ladybugtools_toolkit.plot.utilities import figure_to_base64

        epw = EPW(epw_file)
        data_type_key = data_type_key.replace("_"," ")
        coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])
        fig = heatmap(collection_to_series(coll), cmap=colour_map).get_figure()
        if save_path == None:
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
        "-dtk",
        "--data_type_key",
        help="Key in EPW data to create a plot from.",
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

    args = parser.parse_args()
    main(args.epw_file, args.data_type_key, args.colour_map, args.save_path)
