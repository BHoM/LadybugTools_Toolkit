"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path
import json
import matplotlib

def heatmap(epw_file: str, data_type_key: str, colour_map: str, return_file: str, save_path:str = None) -> None:
    """Create a CSV file version of an EPW."""
    try:
        from ladybug.epw import EPW
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybugtools_toolkit.plot._heatmap import heatmap
        from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
        from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
        from ladybugtools_toolkit.ladybug_extension.epw import wet_bulb_temperature
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt

        if colour_map not in plt.colormaps():
            colour_map = "YlGnBu"


        epw = EPW(epw_file)
        
        if data_type_key == "Wet Bulb Temperature":
            coll = wet_bulb_temperature(epw)
        else:
            coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])
        
        fig = heatmap(collection_to_series(coll), cmap=colour_map).get_figure()

        return_dict = {}

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path
        
        return_dict["data"] = collection_metadata(coll)

        with open(return_file, "w") as rtn:
            rtn.write(json.dumps(return_dict, default=str))
        
        print(return_file)
            

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
        required=True,
        )
    parser.add_argument(
        "-r",
        "--return_file",
        help="json file to write return data to.",
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
    matplotlib.use("Agg")
    heatmap(args.epw_file, args.data_type_key, args.colour_map, args.return_file, args.save_path)
