"""Method to wrap for conversion of IES GEM to HBJSON file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
from pathlib import Path


def main(gem_file: str) -> None:
    """Create a HBJSON file from an IES GEM file."""
    try:
        from honeybee_ies.reader import model_from_ies

        gem_file_path = Path(gem_file)
        model = model_from_ies(gem_file_path.as_posix())
        model.to_hbjson(folder=gem_file_path.parent.as_posix(), name=gem_file_path.stem)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given a GEM file path, convert to a HBJSON file.")
    )
    parser.add_argument(
        "-g",
        "--gem_file",
        help="The GEM file to convert to HBJSON.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.gem_file)
