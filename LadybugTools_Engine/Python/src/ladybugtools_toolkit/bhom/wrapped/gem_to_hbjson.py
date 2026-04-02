"""Method to wrap for conversion of IES GEM to HBJSON file."""
# pylint: disable=C0415,E0401,W0703
import argparse
import sys
import traceback
from pathlib import Path
import tempfile
import json
from honeybee_ies.reader import model_from_ies
from ..logger import CONSOLE_LOGGER

PARSER = argparse.ArgumentParser(
    description=("Given a GEM file path, convert to a HBJSON file.")
)
PARSER.add_argument(
    "-g",
    "--gem_file",
    help="The GEM file to convert to HBJSON.",
    type=str,
    required=True,
)

def gem_to_hbjson(gem_file: str) -> None:
    """Create a HBJSON file from an IES GEM file."""
    try:
        file_path = None
        is_temp = False

        try:
            file_path = Path(gem_file)
        except:
            CONSOLE_LOGGER.info("gem file provided was not a path, trying read as string...")

        if file_path is None or not file_path.is_file():
            is_temp = True
            file_path = Path(tempfile.gettempdir()) / "tempGemFile.gem"
            file_path.write_text(gem_file)
        
        model = model_from_ies(file_path.as_posix())
        model_dict = model.to_dict()

        if is_temp:
            file_path.unlink()

        return json.dumps(model_dict)
    except Exception:
        CONSOLE_LOGGER.error("HBJSON file could not be created.", exc_info=1)
        return traceback.format_exc()


if __name__ == "__main__":
    args = PARSER.parse_args()
    gem_to_hbjson(args.gem_file)
