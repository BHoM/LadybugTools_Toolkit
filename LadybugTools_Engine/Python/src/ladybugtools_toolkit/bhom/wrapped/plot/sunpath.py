﻿"""Method to wrap creation of sunpath plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path
import matplotlib

def sun_path(epw_file, analysis_period, size, return_file: str, save_path):
    try:
        from ladybugtools_toolkit.plot._sunpath import sunpath
        from ladybug.epw import EPW, AnalysisPeriod
        from ladybug.datacollection import HourlyContinuousCollection
        from ladybug.sunpath import Sunpath
        from ladybugtools_toolkit.bhom.wrapped.metadata.sunpath_metadata import sunpath_metadata
        from ladybugtools_toolkit.plot.utilities import figure_to_base64
        import matplotlib.pyplot as plt
        from pathlib import Path
        import json

        analysis_period = AnalysisPeriod.from_dict(json.loads(analysis_period))
        epw = EPW(epw_file)
        fig = sunpath(
            location=epw.location, 
            analysis_period=analysis_period, 
            sun_size=size, 
        ).get_figure()

        return_dict = {"data": sunpath_metadata(Sunpath.from_location(epw.location))}

        if save_path is None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            return_dict["figure"] = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            return_dict["figure"] = save_path
        
        with open(return_file, "w") as rtn:
            rtn.write(json.dumps(return_dict, default=str))

        print(return_file)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, create a plot of its' sun path"
        )
    )
    parser.add_argument(
        "-e",
        "--epw_file",
        help="The EPW file to extract a sun path plot from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--size",
        help="Size of the sun",
        type=float,
        required=True,
        )
    parser.add_argument(
        "-ap",
        "--analysis_period",
        help="Analysis perioderiod of the sun path",
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
    sun_path(args.epw_file, args.analysis_period, args.size, args.return_file, args.save_path)