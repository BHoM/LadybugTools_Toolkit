import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from ladybug.sql import SQLiteResult
from ladybug_extension.datacollection import to_series


def _load_pts_file(pts_file: Union[str, Path]) -> pd.DataFrame:
    pts_file = Path(pts_file)
    df = pd.read_csv(
        pts_file, header=None, names=["x", "y", "z", "vx", "vy", "vz"], sep="\s+"
    )
    df.columns = pd.MultiIndex.from_product([[pts_file.stem], df.columns])
    return df


def _load_pts_files(pts_files: List[Union[str, Path]]) -> pd.DataFrame:

    return pd.concat([_load_pts_file(i) for i in pts_files], axis=1)


def load_pts(pts_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    if isinstance(pts_files, (str, Path)):
        pts_files = [pts_files]
    return _load_pts_files(pts_files)


def _load_res_file(res_file: Union[str, Path]) -> pd.Series:
    res_file = Path(res_file)
    series = pd.read_csv(res_file, header=None, sep="\s+").squeeze()
    series.name = res_file.stem
    return series


def _load_res_files(res_files: List[Union[str, Path]]) -> pd.DataFrame:

    return pd.concat([_load_res_file(i) for i in res_files], axis=1)


def load_res(res_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    if isinstance(res_files, (str, Path)):
        res_files = [res_files]
    return _load_res_files(res_files)


def _load_sun_up_hours_file(
    sun_up_hours_file: Union[str, Path], year: int = 2017
) -> pd.DatetimeIndex:

    sun_up_hours_file = Path(sun_up_hours_file)
    float_hours = pd.read_csv(sun_up_hours_file, header=None, index_col=None).squeeze()
    start_date = datetime.datetime(year, 1, 1, 0, 0, 0)
    timedeltas = [datetime.timedelta(hours=i) for i in float_hours]
    index = pd.DatetimeIndex([start_date + i for i in timedeltas])

    return index


def _load_ill_file(ill_file: Union[str, Path]) -> pd.Series:
    ill_file = Path(ill_file)
    sun_up_hours_file = ill_file.parent / "sun-up-hours.txt"
    df = pd.read_csv(ill_file, sep="\s+", header=None, index_col=None).T
    df.columns = pd.MultiIndex.from_product([[ill_file.stem], df.columns])
    df.index = _load_sun_up_hours_file(sun_up_hours_file)
    return df


def _load_ill_files(ill_files: List[Union[str, Path]]) -> pd.DataFrame:

    return pd.concat([_load_ill_file(i) for i in ill_files], axis=1)


def load_ill(ill_files: Union[str, Path, List[Union[str, Path]]]) -> pd.DataFrame:
    if isinstance(ill_files, (str, Path)):
        ill_files = [ill_files]
    return _load_ill_files(ill_files)


def make_annual(df: pd.DataFrame) -> pd.DataFrame:
    year = df.index[0].year
    freq = f"{(df.index[1] - df.index[0]).total_seconds():0.0f}S"
    df2 = pd.DataFrame(
        index=pd.date_range(
            f"{year}-01-01 00:00:00", f"{year + 1}-01-01 00:00:00", freq=freq
        )[:-1]
    )
    df_reindexed = pd.concat([df2, df], axis=1).fillna(0)
    df_reindexed.columns = pd.MultiIndex.from_tuples(df_reindexed.columns)
    return df_reindexed


def load_sql_file(sql_file: Union[str, Path]) -> pd.DataFrame:
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

        headers.append((element, subelement, f"{variable} ({unit})"))
    df = pd.concat(serieses, axis=1)
    df.columns = pd.MultiIndex.from_tuples(headers)
    df.sort_index(axis=1, inplace=True)
    return df
