"""Method to wrap for access to pre-defined materials."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
import json
from ladybugtools_toolkit.external_comfort.material import Materials

PARSER = argparse.ArgumentParser(
    description=(
        "Given a JSON file path, write the pre-defined materials for the External Comfort workflow."
    )
)
PARSER.add_argument(
    "-j",
    "--json_file",
    help="The JSON file to write material objects into.",
    type=str,
    required=True,
)

def get_material(json_file: str) -> None:
    """Create a file containing all default materials."""
    try:
        json_str = json.dumps([material.value.to_dict() for material in Materials])

        with open(json_file, "w") as f:
            f.write(json_str)

        return json_str

    except Exception as e:
        return traceback.format_exc()


if __name__ == "__main__":
    args = PARSER.parse_args()
    get_material(args.json_file)
