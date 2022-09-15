from pathlib import Path
from typing import List, Union

from ladybug._datacollectionbase import BaseCollection
from ladybug.datautil import collections_to_csv


from python_toolkit.bhom.analytics import analytics


@analytics
def to_csv(collections: List[BaseCollection], csv_path: Union[Path, str]) -> Path:
    """Save Ladybug BaseCollection-like objects into a CSV file.

    Args:
        collection (BaseCollection): A Ladybug BaseCollection-like object.
        csv_path (Union[Path, str]): The path to the CSV file.

    Returns:
        Path: The path to the CSV file.

    """

    csv_path = Path(csv_path)

    if not all(isinstance(n, BaseCollection) for n in collections):
        raise ValueError(
            'All elements of the input "collections" must inherit from BaseCollection.'
        )

    return Path(
        collections_to_csv(
            collections,
            folder=csv_path.parent.as_posix(),
            file_name=csv_path.name,
        )
    )
