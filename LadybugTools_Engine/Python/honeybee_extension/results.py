import sys
sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

import datetime
import warnings
from pathlib import Path
from typing import Callable, List, Union

import pandas as pd
from ladybug.sql import SQLiteResult

from ladybug_extension.datacollection import to_series


def _load_files(func: Callable, files: List[Union[str, Path]]) -> pd.DataFrame:
    """Load a set of input files and combine into a DataFrame with filename as header.

    Args:
        input_files (List[Union[str, Path]]): A list of paths to the input files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input files.
    """
    if isinstance(files, (str, Path)):
        files = [files]

    filenames = [Path(i).stem for i in files]
    if len(set(filenames)) != len(filenames):
        err_str = f"There are duplicate filenames in the list of input files for {func.__name__}. This may cause issues when trying to reference specific results sets!"
        warnings.warn(err_str)

    return pd.concat([func(i) for i in files], axis=1).sort_index(axis=1)


def _load_sun_up_hours_file(
    sun_up_hours_file: Union[str, Path], year: int = 2017
) -> pd.DatetimeIndex:
    """Load a HB-Radiance generated sun-up-hours.txt file and return a DatetimeIndex with the data.

    Args:
        sun_up_hours_file (Union[str, Path]): Path to the sun-up-hours.txt file.
        year (int, optional): The year to be used to generate the DatetimeIndex. Defaults to 2017.

    Returns:
        pd.DatetimeIndex: A Pandas DatetimeIndex with the data from the sun-up-hours.txt file.
    """

    sun_up_hours_file = Path(sun_up_hours_file)
    float_hours = pd.read_csv(sun_up_hours_file, header=None, index_col=None).squeeze()
    start_date = datetime.datetime(year, 1, 1, 0, 0, 0)
    timedeltas = [datetime.timedelta(hours=i) for i in float_hours]
    index = pd.DatetimeIndex([start_date + i for i in timedeltas])

    return index


def _make_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame with partial annual data to a DataFrame with annual data.

    Args:
        df (pd.DataFrame): A DataFrame with partially annually indexed data.

    Returns:
        pd.DataFrame: A DataFrame with annually indexed data.
    """
    assert (
        df.index.year.min() == df.index.year.max()
    ), "The DataFrame must be indexed with annual data within a single year."

    year = df.index[0].year
    freq = f"{(df.index[1] - df.index[0]).total_seconds():0.0f}S"
    minutes_of_hour = df.index.minute.unique()
    df2 = pd.DataFrame(
        index=pd.date_range(
            f"{year}-01-01 00:{minutes_of_hour.min()}:00", f"{year + 1}-01-01 00:{minutes_of_hour.min()}:00", freq=freq
        )[:-1]
    )
    df_reindexed = pd.concat([df2, df], axis=1)
    df_reindexed.columns = pd.MultiIndex.from_tuples(df_reindexed.columns)
    return df_reindexed


def _load_pts_file(pts_file: Union[str, Path]) -> pd.DataFrame:
    """Load a Radiance .pts file and return a DataFrame with the data.

    Args:
        pts_file (Union[str, Path]): The path to the Radiance .pts file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .pts file.
    """
    pts_file = Path(pts_file)
    df = pd.read_csv(
        pts_file, header=None, names=["x", "y", "z", "vx", "vy", "vz"], sep="\s+"
    )
    df.columns = pd.MultiIndex.from_product([[pts_file.stem], df.columns])
    return df


def _load_res_file(res_file: Union[str, Path]) -> pd.Series:
    """Load a Radiance .res file and return a DataFrame with the data. NOTE: This also works with daylight metrics files (da, cda, udi, udi_lower and udi_upper).

    Args:
        res_file (Union[str, Path]): The path to the Radiance .res file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .res file.
    """
    res_file = Path(res_file)
    series = pd.read_csv(res_file, header=None, sep="\s+").squeeze()
    series.name = res_file.stem
    return series


def _load_ill_file(ill_file: Union[str, Path]) -> pd.Series:
    """Load a Radiance .ill file and return a DataFrame with the data.

    Args:
        ill_file (Union[str, Path]): The path to the Radiance .ill file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .ill file.
    """
    ill_file = Path(ill_file)
    sun_up_hours_file = ill_file.parent / "sun-up-hours.txt"
    df = pd.read_csv(ill_file, sep="\s+", header=None, index_col=None).T
    df.columns = pd.MultiIndex.from_product([[ill_file.stem], df.columns])
    df.index = _load_sun_up_hours_file(sun_up_hours_file)
    return df


def _load_sql_file(sql_file: Union[str, Path]) -> pd.DataFrame:
    """Return a DataFrame with hourly values along rows and variables along columns."""

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
            subelement = collection.header.metadata["Surface"]
        elif "System" in collection.header.metadata.keys():
            element = "System"
            subelement = collection.header.metadata["System"]
        elif "Zone" in collection.header.metadata.keys():
            element = "Zone"
            subelement = collection.header.metadata["Zone"]

        headers.append((sql_file.as_posix(), element, subelement, f"{variable} ({unit})"))
    df = pd.concat(serieses, axis=1)
    df.columns = pd.MultiIndex.from_tuples(headers)
    return df


def load_pts(pts_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .pts file, or list of Radiance .pts files and return a combined DataFrame with the data.

    Args:
        pts_files (Union[str, Path, List[Union[str, Path]]]): A single .pts file, or a list of .pts files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .pts files.
    """
    return _load_files(_load_pts_file, pts_files)


def load_res(res_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .res file, or list of Radiance .res files and return a combined DataFrame with the data. NOTE: This also works with daylight metrics files (da, cda, udi, udi_lower and udi_upper).

    Args:
        res_files (Union[str, Path, List[Union[str, Path]]]): A single .res file, or a list of .res files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .res files.
    """
    return _load_files(_load_res_file, res_files)


def load_ill(ill_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single Radiance .ill file, or list of Radiance .ill files and return a combined DataFrame with the data.

    Args:
        ill_files (Union[str, Path, List[Union[str, Path]]]): A single .ill file, or a list of .ill files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .ill files.
    """
    return _load_files(_load_ill_file, ill_files)


def load_sql(sql_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    """Load a single EnergyPlus .sql file, or list of EnergyPlus .sql files and return a combined DataFrame with the data.

    Args:
        sql_files (Union[str, Path, List[Union[str, Path]]]): A single .sql file, or a list of .sql files.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the input .sql files.
    """
    return _load_files(_load_sql_file, sql_files)

if __name__ == "__main__":
    pass