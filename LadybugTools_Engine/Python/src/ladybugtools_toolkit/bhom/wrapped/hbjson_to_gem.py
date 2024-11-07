"""Method to wrap for conversion of HBJSON to GEM file."""

# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def main(hbjson_file: str) -> None:
    """Create an IES GEM file from an HBJSON file."""
    try:
        from honeybee.model import Model
        from honeybee_ies.writer import model_to_ies

        hbjson_file_path = Path(hbjson_file)
        model = Model.from_hbjson(hbjson_file_path.as_posix())
        model_to_ies(
            model,
            folder=hbjson_file_path.parent.as_posix(),
            name=hbjson_file_path.stem)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given an HBJSON file path, convert to a GEM file.")
    )
    parser.add_argument(
        "-j",
        "--hbjson_file",
        help="The HBJSON file to convert to GEM.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.hbjson_file)
