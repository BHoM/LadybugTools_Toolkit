"""Method to wrap for access to pre-defined typologies."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
import json


def main(json_file: str) -> None:
    """Create a file containing all default typologies."""
    try:
        from ladybugtools_toolkit.external_comfort.typology import Typologies

        ds = []
        for typ in Typologies:
            ds.append(typ.value.to_dict())
        with open(json_file, "w") as f:
            json.dump(ds, f)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given a JSON file path, write the pre-defined typologies for the External Comfort workflow."
        )
    )
    parser.add_argument(
        "-j",
        "--json_file",
        help="The JSON file to write Typology objects into.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.json_file)
