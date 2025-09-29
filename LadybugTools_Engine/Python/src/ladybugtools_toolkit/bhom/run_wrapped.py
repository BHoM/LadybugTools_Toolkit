import sys
import argparse
from pathlib import Path
from typing import List


import matplotlib
matplotlib.use("Agg") #use a gui-less backend to avoid memory leaking figures

#big import list that covers all methods in bhom/wrapped
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
from ladybugtools_toolkit.bhom.wrapped.plot.utci_heatmap import utci_heatmap

#import methods and parsers
from ladybugtools_toolkit.bhom.wrapped.plot.walkability_heatmap import PARSER as walkability_heatmap_parser, walkability_heatmap
from ladybugtools_toolkit.bhom.wrapped.plot.windrose import PARSER as windrose_parser, windrose
from ladybugtools_toolkit.bhom.wrapped.plot.directional_solar_radiation import PARSER as directional_solar_radiation_parser, directional_solar_radiation
from ladybugtools_toolkit.bhom.wrapped.plot.diurnal import PARSER as diurnal_parser, diurnal
from ladybugtools_toolkit.bhom.wrapped.plot.facade_condensation_risk_chart import PARSER as facade_condensation_risk_chart_parser, facade_condensation_risk_chart
from ladybugtools_toolkit.bhom.wrapped.plot.facade_condensation_risk_heatmap import PARSER as facade_condensation_risk_heatmap_parser, facade_condensation_risk_heatmap
from ladybugtools_toolkit.bhom.wrapped.plot.heatmap import PARSER as heatmap_parser, heatmap
from ladybugtools_toolkit.bhom.wrapped.plot.sunpath import PARSER as sunpath_parser, sunpath
from ladybugtools_toolkit.bhom.wrapped.plot.utci_heatmap import PARSER as utci_heatmap_parser, utci_heatmap
from ladybugtools_toolkit.bhom.wrapped.epw_to_csv import PARSER as epw_to_csv_parser, epw_to_csv
from ladybugtools_toolkit.bhom.wrapped.gem_to_hbjson import PARSER as gem_to_hbjson_parser, gem_to_hbjson
from ladybugtools_toolkit.bhom.wrapped.get_material import PARSER as get_material_parser, get_material
from ladybugtools_toolkit.bhom.wrapped.get_typology import PARSER as get_typology_parser, get_typology
from ladybugtools_toolkit.bhom.wrapped.hbjson_to_gem import PARSER as hbjson_to_gem_parser, hbjson_to_gem

from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
import matplotlib.pyplot as plt
import numpy as np
import json

#dictionary containing all the parsers for bhom/wrapped commands
PARSERS = {
    "plot/walkability_heatmap": (walkability_heatmap_parser, walkability_heatmap),
    "plot/windrose": (windrose_parser, windrose),
    "plot/directional_solar_radiation": (directional_solar_radiation_parser, directional_solar_radiation),
    "plot/diurnal": (diurnal_parser, diurnal),
    "plot/facade_condensation_risk_chart": (facade_condensation_risk_chart_parser, facade_condensation_risk_chart),
    "plot/facade_condensation_risk_heatmap": (facade_condensation_risk_heatmap_parser, facade_condensation_risk_heatmap),
    "plot/heatmap": (heatmap_parser, heatmap),
    "plot/sunpath": (sunpath_parser, sunpath),
    "plot/utci_heatmap": (utci_heatmap_parser, utci_heatmap),
    "epw_to_csv": (epw_to_csv_parser, epw_to_csv),
    "gem_to_hbjson": (gem_to_hbjson_parser, gem_to_hbjson),
    "get_material": (get_material_parser, get_material),
    "get_typology": (get_typology_parser, get_typology),
    "hbjson_to_gem": (hbjson_to_gem_parser, hbjson_to_gem),
}

def resolve(data: List[str], epw_folder: Path = Path("C:/epws")) -> str:
    """Parses the given data (that looks like sys.argv[1:]), and gets the command arg which is then used to get the parser for that command,
    parse the rest of the args and finally run the command, then return the output of those commands.
    """
    #parse data as args
    command_parser = argparse.ArgumentParser(description="Command parser")
    command_parser.add_argument("-command", "--command")
    command_arg, unknown_args = command_parser.parse_known_args(data)

    parser_function = PARSERS[command_arg.command]
    args = vars(parser_function[0].parse_args(unknown_args))

    if "epw_file" in args:
        #check if the epw file exists, if not prepend the epw_folder and try to run
        epw = Path(args["epw_file"])
        if not epw.exists():
            epw = epw_folder / epw.name
            args["epw_file"] = str(epw)

    ret = parser_function[1](**args) 
    return ret #gets the function for the requested command, and runs it with arguments parsed with the desired parser.

def deconstruct(data: str) -> List[str]:
    splitted = data.split(";", 3)
    arglen = int(splitted[0])
    filelen = int(splitted[1])

    args:List[str] = json.loads(splitted[2][0:arglen])

    if filelen > 0:
        file = splitted[2][arglen:arglen+filelen]
        args.append("-in")
        args.append(file)

    return args

def run_wrapped(args):
    res = resolve(args)
    if res == "":
        sys.exit(1)
    return res

if __name__ == "__main__":
    #this is run if there is no server
    print(run_wrapped(sys.argv[1:]))