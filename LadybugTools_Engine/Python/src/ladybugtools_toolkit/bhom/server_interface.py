import socket
import sys
import threading
import traceback
import shlex
import argparse
import traceback
from pathlib import Path
from typing import List

import time
start = time.time()

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
from ladybugtools_toolkit.bhom.wrapped.get_material import PARSER as get_material_parser, get_material
from ladybugtools_toolkit.bhom.wrapped.get_typology import PARSER as get_typology_parser, get_typology

from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
import matplotlib.pyplot as plt
import numpy as np
import json
from ladybugtools_toolkit.bhom import HOST, PORT

#dictionary containing all the parsers for bhom/wrapped commands
PARSERS = {
    "plot/walkability_heatmap": walkability_heatmap_parser,
    "plot/windrose": windrose_parser,
    "plot/directional_solar_radiation": directional_solar_radiation_parser,
    "plot/diurnal": diurnal_parser,
    "plot/facade_condensation_risk_chart": facade_condensation_risk_chart_parser,
    "plot/facade_condensation_risk_heatmap": facade_condensation_risk_heatmap_parser,
    "plot/heatmap": heatmap_parser,
    "plot/sunpath": sunpath_parser,
    "plot/utci_heatmap": utci_heatmap_parser,
    "get_material": get_material_parser,
    "get_typology": get_typology_parser,
}

def resolve(data: List[str]) -> str:
    """Parses the given data (that looks like sys.argv[1:]), and gets the command arg which is then used to get the parser for that command,
    parse the rest of the args and finally run the command, then return the output of those commands.
    """
    #parse data as args
    command_parser = argparse.ArgumentParser(description="Command parser")
    command_parser.add_argument("-command", "--command")
    command_arg, unknown_args = command_parser.parse_known_args(data)
    kwargs = vars(PARSERS[command_arg.command].parse_args(unknown_args))

    match command_arg.command:
        case "plot/directional_solar_radiation":
            return directional_solar_radiation(**kwargs)
        case "plot/utci_heatmap":
            return utci_heatmap(**kwargs)
        case "plot/walkability_heatmap":
            return walkability_heatmap(**kwargs)
        case "plot/windrose":
            return windrose(**kwargs)
        case "plot/directional_solar_radiation":
            return directional_solar_radiation(**kwargs)
        case "plot/diurnal":
            return diurnal(**kwargs)
        case "plot/facade_condensation_risk_chart":
            return facade_condensation_risk_chart(**kwargs)
        case "plot/facade_condensation_risk_heatmap":
            return facade_condensation_risk_heatmap(**kwargs)
        case "plot/heatmap":
            return heatmap(**kwargs)
        case "plot/sunpath":
            return sunpath(**kwargs)
        case "plot/utci_heatmap":
            return utci_heatmap(**kwargs)

def socket_handler(client_socket, addr):
    print("connection received:", addr)
    with client_socket:
        data = client_socket.recv(1024)
        if not data:
            return
        #parse data as args
        args = json.loads(data.decode())
        print("received args:", args)
        result = resolve(args)
        client_socket.sendall(result.encode())
    print("connection closed:", addr)

def server(host: str = HOST, port: int = PORT):
    """The "server" socket for interaction between bhom c# and python. This could be used with a socket connection from any other language.
    The server accepts data in json format that looks like arguments from the command line, and sends those arguments to the parser to allow them to run,
    then returns the output data and then closes the connection. This method allows for a persistent python interpreter to avoid reloading libraries every time a script is run.

    Args:
        host (str):
            The host name that the socket listens on. defaults to 127.0.0.1
        port (int):
            The port that the server accepts data from. defaults to 5999
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(5)
    s.settimeout(1)
    print(f"listening on {host} with port {port}")

    #main listener loop
    while True:
        try:
            client_socket, addr = s.accept()
            threading.Thread(target=socket_handler, args=(client_socket, addr), daemon=True).start() #daemon=True, so that the thread exits if the main program exits (i.e. due to keyboard interrupt)
        except socket.timeout: #timeouts exist to allow keyboard interrupts to close the program properly.
            pass
        except KeyboardInterrupt:
            sys.exit(1)
        except Exception as ex:
            print(traceback.format_exc())
            pass

end = time.time()

if __name__ == "__main__":
    #run with preset host and port (127.0.0.1 on 5999)
    print("module load took", str(end - start), "seconds")
    server()
    print("closing app")