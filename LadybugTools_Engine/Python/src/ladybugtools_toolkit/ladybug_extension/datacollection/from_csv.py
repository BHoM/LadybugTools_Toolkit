from pathlib import Path
from typing import List, Union

from ladybug._datacollectionbase import BaseCollection
from ladybug.datautil import collections_from_csv


from ladybugtools_toolkit import analytics


@analytics
def from_csv(csv_path: Union[Path, str]) -> List[BaseCollection]:
    """Load a CSV containing serialised Ladybug BaseCollection-like objects.

    Args:
        csv_path (Union[Path, str]): The path to the CSV file.

    Returns:
        List[BaseCollection]: A list of Ladybug BaseCollection-like object.

    """

    csv_path = Path(csv_path)

    if not csv_path.suffix == ".csv":
        raise ValueError("The target file must be a *.csv file.")

    return collections_from_csv(csv_path.as_posix())
