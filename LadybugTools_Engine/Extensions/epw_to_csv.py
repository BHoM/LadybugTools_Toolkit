#!/usr/bin/env python

from ladybug.epw import EPW
import json
import sys
from pathlib import Path

sys.path.append(r"C:\ProgramData\BHoM\Extensions")
from LadybugTools.epw_to_dataframe import epw_to_dataframe

def epw_to_csv(epw_file: str):
    """Convert an EPW into a CSV, including a few extra fields."""

    epw_path = Path(epw_file)
    csv_path = epw_path.with_suffix(".csv")
    epw = EPW(epw_file)
    df = epw_to_dataframe(epw)
    df.to_csv(csv_path)
    return csv_path


if __name__ == "__main__":
    print(epw_to_csv(sys.argv[1]))
