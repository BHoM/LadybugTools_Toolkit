from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.spatial.load.points import points
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric import (
    SpatialMetric,
)
from ladybugtools_toolkit.external_comfort.spatial.metric.spatial_metric_filepath import (
    spatial_metric_filepath,
)
from ladybugtools_toolkit.external_comfort.spatial.moisture_distribution.load_moisture_sources import (
    load_moisture_sources,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


from python_toolkit.bhom.analytics import analytics


@analytics
def evap_clg_magnitude(simulation_directory: Path, epw: EPW) -> pd.DataFrame:
    """Calculate the magnitude of evaporative cooling across a spatial case.

    Args:
        simulation_directory (Path):
            The directory containing simulation results.
        epw (EPW):
            The associated EPW file.

    Returns:
        pd.DataFrame: A spatial matrix of time/pt containing evaporative cooling effective.
    """

    metric = SpatialMetric.EVAP_CLG

    moisture_path = spatial_metric_filepath(simulation_directory, metric)

    if moisture_path.exists():
        print(f"[{simulation_directory.name}] - Loading {metric.value}")
        moisture_df = pd.read_parquet(moisture_path)
        moisture_df.columns = moisture_df.columns.astype(int)
        return moisture_df

    print(f"[{simulation_directory.name}] - Generating {metric.value}")

    # create an index to attribute calculated values
    idx = to_series(epw.dry_bulb_temperature).index

    # load moisture sources
    moisture_sources = load_moisture_sources(simulation_directory)

    # load spatial points in list of [[X, Y], [X, Y], [X, Y]]
    spatial_points = points(simulation_directory)[["x", "y"]].values

    # get moisture matrices per moisture source, and resultant matrix
    moisture_df = pd.DataFrame(
        np.amax(
            [
                i.spatial_moisture(
                    spatial_points, epw, simulation_directory=simulation_directory
                )
                for i in moisture_sources
            ],
            axis=0,
        ),
        index=idx,
    )

    # save matrix to file
    moisture_df.columns = moisture_df.columns.astype(str)
    moisture_df.to_parquet(moisture_path)

    return moisture_df
