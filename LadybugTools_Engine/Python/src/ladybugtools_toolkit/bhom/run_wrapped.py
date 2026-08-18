import sys
import argparse
from pathlib import Path
from typing import List, Callable
import json

from python_toolkit.bhom.decorators import bhom_wrapper

import matplotlib
matplotlib.use("Agg") #use a gui-less backend to avoid memory leaking figures

#big import list that covers all methods in bhom/wrapped
from . import wrapped
from python_toolkit.bhom import wrapped

COMMAND_PARSER = argparse.ArgumentParser(description="argument parser for commands.")
COMMAND_PARSER.add_argument("-command", "--command")
COMMAND_PARSER.add_argument("-in", "--input_json")

def resolve(data: List[str], epw_folder: Path = Path("C:/epws")) -> str:
    """Parses the given data (that looks like sys.argv[1:]), and gets the command arg which is an identifier for the command which is requested,
    and the input json string (or file) to be given to the BHoMJSONDecoder wrapped method.

    Also if the given epw file doesn't exist, assume that it is a file name and append it to the epw folder as a backup.
    """
    #parse data as args
    command_parser = argparse.ArgumentParser(description="argument parser for commands.")
    command_parser.add_argument("-command", "--command")
    command_parser.add_argument("-in", "--input_json")
    command_parser.add_argument("-e", "--epw_file", required=False)
    command_args, unknown_args = command_parser.parse_known_args(data)

    if command_args.epw_file is not None:
        #check if the epw file exists, if not prepend the epw_folder and try to run
        epw = Path(command_args.epw_file)
        if not epw.exists():
            epw = epw_folder / epw.name
            command_args.epw_file = str(epw)

    method = bhom_wrapper.get_registered_method(command_args.command)

    ret = method(epw_file = command_args.epw_file, __input_json__ = command_args.input_json) 
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