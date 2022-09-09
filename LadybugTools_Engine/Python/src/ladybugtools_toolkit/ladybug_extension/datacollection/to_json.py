from pathlib import Path
from typing import List, Union

from ladybug._datacollectionbase import BaseCollection
from ladybug.datautil import collections_to_json


def to_json(
    collections: List[BaseCollection], json_path: Union[Path, str], indent: int = None
) -> Path:
    """Save Ladybug BaseCollection-like objects into a JSON file.

    Args:
        collection (BaseCollection): A Ladybug BaseCollection-like object.
        json_path (Union[Path, str]): The path to the JSON file.
        indent (str, optional): The indentation to use in the resulting JSON file. Defaults to None.

    Returns:
        Path: The path to the JSON file.

    """

    json_path = Path(json_path)

    if not all(isinstance(n, BaseCollection) for n in collections):
        raise ValueError(
            'All elements of the input "collections" must inherit from BaseCollection.'
        )

    if not json_path.suffix == ".json":
        raise ValueError("The target file must be a *.json file.")

    return Path(
        collections_to_json(
            collections,
            folder=json_path.parent.as_posix(),
            file_name=json_path.name,
            indent=indent,
        )
    )
