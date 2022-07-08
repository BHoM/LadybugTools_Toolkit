import json

from pathlib import Path
from typing import Union

from ladybug._datacollectionbase import BaseCollection


def to_json(collection: BaseCollection, json_path: Union[Path, str]) -> str:
    """Save a Ladybug BaseCollection-like object into a JSON file.

    Args:
        collection (BaseCollection): A Ladybug BaseCollection-like object.
        json_path (Union[Path, str]): The path to the JSON file.

    Returns:
        str: The path to the JSON file.

    """

    json_path = Path(json_path)

    if not json_path.suffix == ".json":
        raise ValueError("The target file must be a *.json file.")

    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)
    d = collection.to_dict()
    with open(json_path, "w") as f:
        f.write(json.dumps(d))

    return str(json_path)
