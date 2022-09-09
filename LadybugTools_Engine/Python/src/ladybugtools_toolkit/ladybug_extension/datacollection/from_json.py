from pathlib import Path
from typing import List, Union

from ladybug._datacollectionbase import BaseCollection
from ladybug.datautil import collections_from_json


def from_json(json_path: Union[Path, str]) -> List[BaseCollection]:
    """Load a JSON containing serialised Ladybug BaseCollection-like objects.

    Args:
        json_path (Union[Path, str]): The path to the JSON file.

    Returns:
        List[BaseCollection]: A list of Ladybug BaseCollection-like object.

    """

    json_path = Path(json_path)

    if not json_path.suffix == ".json":
        raise ValueError("The target file must be a *.json file.")

    return collections_from_json(json_path.as_posix())
