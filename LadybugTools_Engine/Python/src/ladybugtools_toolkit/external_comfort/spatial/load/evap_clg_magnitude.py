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

    evap_clg_path = spatial_metric_filepath(simulation_directory, metric)

    if evap_clg_path.exists():
        print(f"- Loading {metric.value} from {simulation_directory.name}")
        return pd.read_hdf(evap_clg_path, "df")

    print(f"- Generating {metric.value} for {simulation_directory.name}")

    # create an index to attribute calculated values
    idx = to_series(epw.dry_bulb_temperature).index

    # load moisture sources
    moisture_sources = load_moisture_sources(simulation_directory)

    # load spatial points in list of [[X, Y], [X, Y], [X, Y]]
    spatial_points = (
        points(simulation_directory).droplevel(0, axis=1)[["x", "y"]].values
    )

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
    moisture_df.to_hdf(evap_clg_path, "df", complib="blosc", complevel=9)

    return moisture_df
