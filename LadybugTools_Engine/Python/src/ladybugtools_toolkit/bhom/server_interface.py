import socket
import ssl
import sys
import threading
import traceback
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
    "plot/walkability_heatmap": (walkability_heatmap_parser, walkability_heatmap),
    "plot/windrose": (windrose_parser, windrose),
    "plot/directional_solar_radiation": (directional_solar_radiation_parser, directional_solar_radiation),
    "plot/diurnal": (diurnal_parser, diurnal),
    "plot/facade_condensation_risk_chart": (facade_condensation_risk_chart_parser, facade_condensation_risk_chart),
    "plot/facade_condensation_risk_heatmap": (facade_condensation_risk_heatmap_parser, facade_condensation_risk_heatmap),
    "plot/heatmap": (heatmap_parser, heatmap),
    "plot/sunpath": (sunpath_parser, sunpath),
    "plot/utci_heatmap": (utci_heatmap_parser, utci_heatmap),
    "get_material": (get_material_parser, get_material),
    "get_typology": (get_typology_parser, get_typology),
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

def get_byte_counts(data: bytes):
    lengths = data.decode()
    print("Receivd args and file lengths:", lengths)
    s = lengths.split(';')
    arg_byte_length = int(s[0])
    file_byte_length = int(s[1])
    return arg_byte_length, file_byte_length
    

def socket_handler(client_socket: ssl.SSLSocket, addr, epw_folder: Path):
    print("connection received:", addr)

    with client_socket:
        recvd = ""
        data = client_socket.recv(1024)

        if not data: #if client doesn't send anything and closes the connection early, return early.
            print("connection closed early:", addr)
            return

        arg_bytes, file_bytes = get_byte_counts(data)

        args_recvd = ""

        while arg_bytes > 0:
            data = client_socket.recv(min([arg_bytes, 1024]))
            args_recvd += data.decode()
            arg_bytes -= len(data)

        file_recvd = ""

        while file_bytes > 0:
            data = client_socket.recv(min([file_bytes, 1024]))
            file_recvd += data.decode()
            file_bytes -= len(data)

        #parse data as args
        print(recvd)
        args = json.loads(args_recvd)
        print("received args:", args)
        if file_recvd != "":
            print("Adding in file to args")
            args.append("-in")
            args.append(file_recvd)

        result = resolve(args, epw_folder)
        client_socket.sendall(result.encode())

    print("connection closed:", addr)

def server(host: str = HOST, port: int = PORT, certs: List[str] = [], epw_folder: Path = "C:/epws"):
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

    context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
    for cert in certs:
        context.load_cert_chain(cert)

    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3

    #main listener loop
    while True:
        try:
            client_socket, addr = s.accept()
            print(addr)
            ssock = context.wrap_socket(client_socket, server_side=True, do_handshake_on_connect=True)
            threading.Thread(target=socket_handler, args=(ssock, addr, epw_folder), daemon=True).start() #daemon=True, so that the thread exits if the main program exits (i.e. due to keyboard interrupt)
        except socket.timeout: #timeouts exist to allow keyboard interrupts to close the program properly.
            pass
        except KeyboardInterrupt:
            sys.exit(1)
        except Exception as ex:
            print(traceback.format_exc())
            pass

end = time.time()

if __name__ == "__main__":
    args = sys.argv[1:]
    argparser = argparse.ArgumentParser(prog="LBT Server")
    argparser.add_argument(
        "-i",
        "--host",
        help=f"interface/host address to use, defaults to {HOST}",
        type=str,
        required=False,
        default=HOST
        )
    argparser.add_argument(
        "-p",
        "--port",
        help = f"port number to use, defaults to {PORT}",
        type=int,
        required=False,
        default=PORT
        )
    argparser.add_argument(
        "-cert",
        "--cert",
        help="Path to certificate file(s) to use",
        action="append",
        required=True
        )
    argparser.add_argument(
        "-e",
        "--default_epw",
        help="Path to folder where epws are located by default, used to help resolve epw files.",
        required=False,
        type=Path,
        default=Path("C:/epws")
        )
    print("module load took", str(end - start), "seconds")
    ns = argparser.parse_args()
    print(ns.cert)
    server(ns.host, ns.port, ns.cert, ns.default_epw)
    print("closing app")