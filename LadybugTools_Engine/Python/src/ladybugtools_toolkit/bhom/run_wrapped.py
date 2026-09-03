import sys
import argparse
from pathlib import Path
from typing import List, Callable
import json

from python_toolkit.bhom.decorators import bhom_wrapper

import matplotlib
matplotlib.use("Agg") #use a gui-less backend to avoid memory leaking figures

#big import list that covers all methods in bhom/wrapped
from python_toolkit.bhom import wrapped
from ladybugtools_toolkit.bhom import wrapped

COMMAND_PARSER = argparse.ArgumentParser(description="argument parser for commands.")
COMMAND_PARSER.add_argument("-command", "--command")
COMMAND_PARSER.add_argument("-in", "--input_json")

def resolve(data: List[str], epw_locator: Callable[[str], str] = lambda e: e) -> str:
    """Parses the given data (that looks like sys.argv[1:]), and gets the command arg which is an identifier for the command which is requested,
    and the input json string (or file) to be given to the BHoMJSONDecoder wrapped method.

    epw_locator: callable directly sent to methods in kwargs so that epws can be "located"
        i.e. if the parent directory might not be valid for instance in the context of a web server
        this can replace the parent directory with the expected location for all epw files
        Defaults to `lambda e: e` (don't change the location)

    """
    #parse data as args
    command_args, unknown_args = COMMAND_PARSER.parse_known_args(data)

    method = bhom_wrapper.get_registered_method(command_args.command)

    ret = method(__input_json__ = command_args.input_json, epw_locator = epw_locator) 
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