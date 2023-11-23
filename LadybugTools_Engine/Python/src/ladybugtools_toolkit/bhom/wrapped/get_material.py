"""Method to wrap for access to pre-defined materials."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback
import json


def main(json_file: str) -> None:
    """Create a file containing all default materials."""
    try:
        from ladybugtools_toolkit.external_comfort.material import Materials
        from ladybugtools_toolkit.bhom.to_bhom import material_to_bhom

        with open(json_file, "w") as f:
            json.dump([material.value.to_dict() for material in Materials], f)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given a JSON file path, write the pre-defined materials for the External Comfort workflow."
        )
    )
    parser.add_argument(
        "-j",
        "--json_file",
        help="The JSON file to write material objects into.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.json_file)
