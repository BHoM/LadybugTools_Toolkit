#!/usr/bin/env python

from ladybug.epw import EPW
import json
import sys

def epw_to_json(epw_file: str):
    """Convert an EPW into a JSON string representation version, according to the Ladybug EPW schema."""

    epw = EPW(epw_file)
    json_str = json.dumps(epw.to_dict())
    return json_str


if __name__ == "__main__":
    print(epw_to_json(sys.argv[1]))
