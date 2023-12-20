"""Method to wrap for access to pre-defined materials."""
# pylint: disable=C0415,E0401,W0703
import argparse
import traceback


def main(json_file: str) -> None:
    """From a json file represention of an ExternalComfort, run the calculation."""
    try:
        from ladybugtools_toolkit.external_comfort._externalcomfortbase import (
            ExternalComfort,
        )

        ec = ExternalComfort.from_file(json_file)
        ec.to_file(json_file)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given a JSON file containing the string represention of a ExternalComfort object, "
            "run all calculations Python-side for that object."
        )
    )
    parser.add_argument(
        "-j",
        "--json_file",
        help="The JSON file to convert into a ExternalComfort object Python-side.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.json_file)
