import warnings
from pathlib import Path
from typing import Union

import pandas as pd
from ladybug.sql import SQLiteResult

from ...ladybug_extension.datacollection.basecollection import to_series


def load_sql_file(sql_file: Union[str, Path]) -> pd.DataFrame:
    """Return a DataFrame with hourly values along rows and variables along columns.

    Args:
        sql_file (Union[str, Path]): The path to the EnergyPlus .sql file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the .sql file.
    """

    sql_file = Path(sql_file)

    sql_obj = SQLiteResult(sql_file.as_posix())

    def _flatten(container):
        for i in container:
            if isinstance(i, (list, tuple)):
                for j in _flatten(i):
                    yield j
            else:
                yield i

    collections = list(
        _flatten(
            [
                sql_obj.data_collections_by_output_name(i)
                for i in sql_obj.available_outputs
            ]
        )
    )

    serieses = []
    headers = []
    for collection in collections:
        serieses.append(to_series(collection))
        variable = collection.header.metadata["type"]
        unit = collection.header.unit

        if "Surface" in collection.header.metadata.keys():
            element = "Surface"
            sub_element = collection.header.metadata["Surface"]
        elif "System" in collection.header.metadata.keys():
            element = "System"
            sub_element = collection.header.metadata["System"]
        elif "Zone" in collection.header.metadata.keys():
            element = "Zone"
            sub_element = collection.header.metadata["Zone"]
        else:
            warnings.warn(f"Could not determine element type for {variable}")
            element = "Unknown"
            sub_element = "Unknown"

        headers.append(
            (sql_file.as_posix(), element, sub_element, f"{variable} ({unit})")
        )
    df = pd.concat(serieses, axis=1)
    df.columns = pd.MultiIndex.from_tuples(headers)
    return df
