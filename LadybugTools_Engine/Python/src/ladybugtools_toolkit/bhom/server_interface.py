import socket
import traceback
import shlex
import argparse
import traceback
from pathlib import Path
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
from ladybugtools_toolkit.bhom.wrapped.plot.utci_heatmap import utci_heatmap
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
import matplotlib.pyplot as plt
import numpy as np
import json
from ladybugtools_toolkit.bhom import HOST, PORT

parsers = {}

parsers["plot/utci_heatmap"] = argparse.ArgumentParser()
parsers["plot/utci_heatmap"].add_argument(
    "-in",
    "--json_file",
    help="helptext",
    type=str,
    required=True,
)
parsers["plot/utci_heatmap"].add_argument(
    "-r",
    "--return_file",
    help="json file to write return data to.",
    type=str,
    required=True,
)
parsers["plot/utci_heatmap"].add_argument(
    "-sp",
    "--save_path",
    help="helptext",
    type=str,
    required=False,
)

def resolve(data) -> str:
    #parse data as args
    parser = argparse.ArgumentParser(description="Test parser")
    parser.add_argument("-command", "--command")
    args, unknown_args = parser.parse_known_args(data)

    match args.command:
        case "plot/utci_heatmap":
            kwargs = vars(parsers[args.command].parse_args(unknown_args))
            return utci_heatmap(**kwargs)


def app():
    conn = None
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    s.settimeout(1)
    print(f"listening on {HOST} with port {PORT}")
    #main listener loop
    conn = None
    while True:
        try:
            conn, addr = s.accept()
            print("connection received:", addr)
            data = conn.recv(1024)
            if not data:
                continue

            #parse data as args
            args = json.loads(data.decode())
            res = resolve(args)
            conn.sendall(res)
            
        except socket.timeout:
            pass
        except KeyboardInterrupt:
            if conn:
                conn.close()
                print("handling keyboard interrupt")
            break
        except Exception as ex:
            print(traceback.format_exc())
        finally:
            if conn:
                conn.close()
                conn = None
                print("closing connection")
            pass

if __name__ == "__main__":
    app()
    print("closing app")