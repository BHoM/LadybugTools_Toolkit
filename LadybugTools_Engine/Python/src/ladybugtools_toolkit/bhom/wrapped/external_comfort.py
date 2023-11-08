"""Method to wrap for access to pre-defined materials."""
# pylint: disable=E0401
import argparse
import traceback

# pylint: enable=E0401

from ladybugtools_toolkit.external_comfort._externalcomfortbase import ExternalComfort


def main(json_file: str) -> None:
    """From a json file represention of an ExternalComfort, run the calculation."""
    try:
        ec = ExternalComfort.from_file(json_file)
        ec.to_file(json_file)

    except Exception as e:  # pylint: disable=W0703
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Given a JSON file containing the string represention of a ExternalComfort object."
        )
    )
    parser.add_argument(
        "-j",
        "--json_file",
        help="The JSON file to convert into a ExternbalComfort object Python-side.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.json_file)
