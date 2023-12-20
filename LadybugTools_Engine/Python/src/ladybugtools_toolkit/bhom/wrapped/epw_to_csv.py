"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def main(epw_file: str, include_additional: bool) -> None:
    """Create a CSV file version of an EPW."""
    try:
        from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe, EPW

        epw = EPW(epw_file)
        df = epw_to_dataframe(epw=epw, include_additional=include_additional)
        df.to_csv(Path(epw_file).with_suffix(".csv"))

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given an EPW file path, convert to CSV with optional inclusion of calculated additional data."
        )
    )
    parser.add_argument(
        "-e",
        "--epw_file",
        help="The EPW file to write as a CSV.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--include_additional",
        help="Whether to include additional calculated data (such as hourly ground temperature, sky temperature, sun position, ...).",
        type=bool,
        required=True,
    )
    args = parser.parse_args()
    main(args.epw_file, args.include_additional)
