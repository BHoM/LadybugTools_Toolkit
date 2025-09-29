"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import sys
import traceback
from pathlib import Path
from ..logger import CONSOLE_LOGGER
from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe, EPW

PARSER = argparse.ArgumentParser(
    description=(
            "Given an EPW file path, convert to CSV with optional inclusion of calculated additional data."
        )
    )
PARSER.add_argument(
    "-e",
    "--epw_file",
    help="The EPW file to write as a CSV.",
    type=str,
    required=True,
)
PARSER.add_argument(
    "-a",
    "--include_additional",
    help="Whether to include additional calculated data (such as hourly ground temperature, sky temperature, sun position, ...).",
    type=bool,
    required=True,
)

def epw_to_csv(epw_file: str, include_additional: bool) -> str:
    """Create a CSV file version of an EPW."""
    try:
        epw = EPW(epw_file)
        df = epw_to_dataframe(epw=epw, include_additional=include_additional)
        csv_str = df.to_csv()
        return csv_str
    except Exception:
        CONSOLE_LOGGER.error("CSV file could not be created.", exc_info=1)
        return ""

if __name__ == "__main__":
    args = PARSER.parse_args()
    epw_to_csv(args.epw_file, args.include_additional)
