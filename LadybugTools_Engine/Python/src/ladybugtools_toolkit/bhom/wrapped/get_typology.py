"""Method to wrap for access to pre-defined typologies."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
import json
from ladybugtools_toolkit.external_comfort.typology import Typologies

PARSER = argparse.ArgumentParser(
    description=(
        "Given a JSON file path, write the pre-defined typologies for the External Comfort workflow."
    )
)
PARSER.add_argument(
    "-j",
    "--json_file",
    help="The JSON file to write Typology objects into.",
    type=str,
    required=True,
)

def get_typology(json_file: str) -> None:
    """Create a file containing all default typologies."""
    try:
        ds = []
        for typ in Typologies:
            ds.append(typ.value.to_dict())
        with open(json_file, "w") as f:
            json.dump(ds, f)

        return json_file

    except Exception as e:
        return traceback.format_exc()


if __name__ == "__main__":
    args = PARSER.parse_args()
    get_typology(args.json_file)
