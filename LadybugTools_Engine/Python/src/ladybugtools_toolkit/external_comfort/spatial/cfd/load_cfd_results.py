import json
from pathlib import Path

import pandas as pd
from ladybugtools_toolkit.external_comfort.spatial.cfd.cfd_directory import (
    cfd_directory,
)
from ladybugtools_toolkit.external_comfort.spatial.cfd.load_cfd_extract import (
    load_cfd_extract,
)


from ladybugtools_toolkit import analytics


@analytics
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
    with open(cfd_dir / "config.json", "r") as fp:
        config = json.load(fp)

    # load each CSV in config, and construct mega-dataframe
    dfs = []
    for result in config:
        # load the wind speed, adn divide by the source wind velocity to get a normalised velocity
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
