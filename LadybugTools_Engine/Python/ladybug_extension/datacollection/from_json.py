import sys

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import json
from pathlib import Path
from typing import Union

from ladybug._datacollectionbase import BaseCollection
from ladybug.datacollection import (HourlyContinuousCollection,
                                    MonthlyCollection)


def from_json(json_path: Union[Path, str]) -> BaseCollection:
    """Load a JSON containing a serialised Ladybug BaseCollection-like object.

    Args:
        json_path (Union[Path, str]): The path to the JSON file.

    Returns:
        BaseCollection: A Ladybug BaseCollection-like object.

    """

    json_path = Path(json_path)

    if not json_path.suffix == ".json":
        raise ValueError("The target file must be a *.json file.")

    if not json_path.exists():
        raise ValueError("The target file does not exist.")

    with open(json_path, "r") as f:
        d = json.load(f)

    try:
        return HourlyContinuousCollection.from_dict(d)
    except Exception as e:
        try:
            return MonthlyCollection.from_dict(d)
        except Exception as e:
            try:
                return BaseCollection.from_dict(d)
            except Exception as e:
                raise e
