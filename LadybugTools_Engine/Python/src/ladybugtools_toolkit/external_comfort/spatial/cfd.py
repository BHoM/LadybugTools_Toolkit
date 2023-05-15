import json
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from scipy.interpolate import interp1d
from tqdm import tqdm

from ...ladybug_extension.datacollection import collection_to_series
from ...ladybug_extension.epw import unique_wind_speed_direction

EXAMPLE_CFD_CONFIG_JSON = """
[
    {
        "pt_velocity_file": "V225.csv",
        "source_velocity": 4.48,
        "wind_direction": 225
    },
    {
        "pt_velocity_file": "V270.csv",
        "source_velocity": 4.94,
        "wind_direction": 270
    },
    {
        "pt_velocity_file": "V315.csv",
        "source_velocity": 4.83,
        "wind_direction": 315
    },
    {
        "pt_velocity_file": "V000.csv",
        "source_velocity": 3.87,
        "wind_direction": 0
    },
    {
        "pt_velocity_file": "V045.csv",
        "source_velocity": 3.565,
        "wind_direction": 45
    },
    {
        "pt_velocity_file": "V090.csv",
        "source_velocity": 3.78,
        "wind_direction": 90
    },
    {
        "pt_velocity_file": "V135.csv",
        "source_velocity": 3.365,
        "wind_direction": 135
    },
    {
        "pt_velocity_file": "V180.csv",
        "source_velocity": 3.16,
        "wind_direction": 180
    }
]
"""


def spatial_wind_speed(simulation_directory: Path, epw: EPW) -> pd.DataFrame:
    """Calculate the temporo-spatial wind speed for a given SpatialComfort case.

    Args:
        simulation_directory (Path):
            The associated simulation directory
        epw (EPW):
            An EPW object

    Returns:
        pd.DataFrame:
            A time indexed, spatial matrix of point-wind-speeds.
    """

    # for each unique wind direction, get the interpolated normalised wind speed
    uniques = unique_wind_speed_direction(epw)
    unique_directions = np.unique(uniques.T[1])

    # get the cfd results to interpolate between
    cfd_results = load_cfd_results(simulation_directory)

    # interpolate to unique directions from known directions
    x = cfd_results.columns
    y = cfd_results.values
    z = pd.DataFrame(interp1d(x, y)(unique_directions), columns=unique_directions)

    # for each unique combo of ws and wd, get the associated pt-ws
    d = {}
    for ws, wd in tqdm(uniques, desc="Getting unique pt-wind-speeds"):
        d[(ws, wd)] = (z[wd] * ws).values

    # for each row of epw, create spatial velocity
    vals = []
    for ws, wd in list(zip(*[epw.wind_speed, epw.wind_direction])):
        vals.append(d[(ws, wd)])
    idx = collection_to_series(epw.wind_direction).index
    vel_df = pd.DataFrame(np.array(vals), index=idx)

    return vel_df


def load_cfd_results(simulation_directory: Path) -> pd.DataFrame:
    """Load CFD files from the given directory

    Args:
        simulation_directory (Path):
            A Spatial Comfort simulation directory.

    Returns:
        pd.DataFrame:
            A DataFrame containing the data from the input files.
    """

    cfd_dir = cfd_directory(simulation_directory)

    # get json file
    with open(cfd_dir / "config.json", "r", encoding="utf-8") as fp:
        config = json.load(fp)

    # load each CSV in config, and construct mega-dataframe
    dfs = []
    for result in config:
        # load the wind speed, and divide by the source wind velocity to get a normalised velocity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfs.append(
                load_cfd_extract(
                    cfd_dir / result["pt_velocity_file"], result["wind_direction"]
                ).drop(columns=["x", "y", "z"])
                / result["source_velocity"]
            )
    df = pd.concat(dfs, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    if (360 not in df.columns) and (0 in df.columns):
        df[360] = df[0]

    if (360 in df.columns) and (0 not in df.columns):
        df[0] = df[360]

    return df.sort_index(axis=1)


def load_cfd_extract(file: Path, velocity_col: str = "VELOCITY") -> pd.DataFrame:
    """Load a file containing an extract from CFX-Post, with point XYZ values,
        and variable for that point.

    Args:
        file (Path):
            The path to the CFX-Post extract file.
        velocity_col (str, optional):
            A string to rename the last column by.

    Returns:
        pd.DataFrame:
            A DataFrame containing the data from the given file.
    """

    with open(file, "r", encoding="utf-8") as fp:
        dat = fp.read()

    df = (
        pd.read_csv(StringIO(dat.split("\n\n")[1]), sep=",", header=[0, 1])
        .reset_index()
        .drop(columns=["level_0"])
    )
    if len(df.columns) != 4:
        raise ValueError("Columns should be of length 4 (x, y, z, velocity)")

    df.columns = ["x", "y", "z", velocity_col]

    # replace any null values with the avg for the whole dataset
    df[velocity_col] = pd.to_numeric(df[velocity_col], errors="coerce")
    df[velocity_col] = df[velocity_col].fillna(df[velocity_col].mean())

    return df


def cfd_directory(simulation_directory: Path) -> Path:
    """Get the CFD directory for a spatial simulation.

    Args:
        simulation_directory (Path):
            The associated simulation directory

    Returns:
        Path:
            The path to the moisture directory
    """

    if (not (simulation_directory / "cfd").exists()) or (
        not (simulation_directory / "cfd" / "config.json").exists()
    ):
        raise FileNotFoundError(
            f'No "cfd" directory exists in {simulation_directory}. For this method to work, '
            + "you need a cfd directory containing a set of csv files extracted from CFD "
            + "simulations of at least 8 wind directions. Values in these files should correspond "
            + "with the wind velocities at the points from the SpatialComfort case being assessed."
            + "\nFor example, the folder should contain 8 CSV files:"
            + '\n    ["./V315.csv", "./V000.csv", "./V045.csv", "./V090.csv", "./V135.csv", '
            + '"./V180.csv", "./V225.csv", "./V270.csv"]'
            + "\n... each containing point-velocities for the points in the SpatialComfort simulation"
            + "\nAdditionally, a JSON config should also be included, which stores the velocity "
            + "applied across the simulations for scaling in the thermal comfort assessment. An "
            + "example config can be found in this modules __init__.py"
        )

    return simulation_directory / "cfd"
